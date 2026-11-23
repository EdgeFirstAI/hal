// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! H.264 sequence and picture parameter sets (ITU-T H.264 §7.3.2.1,
//! §7.3.2.2, Annex E), slice headers (§7.3.3) and picture order count
//! (§8.2.1).
//!
//! Each parser takes one NAL unit with its header byte and without a start
//! code. Input that is malformed or ends early is an error.

use std::collections::HashMap;

use edgefirst_tensor::Region;

use super::h264::{NAL_IDR, NAL_PPS, NAL_SPS};

/// `nal_unit_type` of a coded slice of a non-IDR picture.
const NAL_SLICE: u8 = 1;

/// Profiles whose SPS carries `chroma_format_idc` and scaling lists.
const HIGH_PROFILES: [u8; 13] = [100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135];

/// Profiles that are intra-only when `constraint_set3_flag` is set (§E.2.1).
const INTRA_PROFILES: [u8; 6] = [44, 86, 100, 110, 122, 244];

/// The largest `MaxFS` of Table A-1, in macroblocks.
const MAX_FRAME_MBS: u64 = 139_264;

const MAX_DPB_FRAMES: usize = 16;

const SLICE_P: u32 = 0;
const SLICE_B: u32 = 1;
const SLICE_SP: u32 = 3;

const TRUNCATED: &str = "H.264 header ends early";

/// The RBSP of a NAL unit: `nal` with every emulation prevention byte
/// (§7.4.1) removed.
pub(crate) fn unescape(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal.len());
    let mut zeros = 0;
    for &b in nal {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        out.push(b);
    }
    out
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn flag(&mut self) -> Result<bool, String> {
        let byte = self.data.get(self.pos / 8).ok_or(TRUNCATED)?;
        let bit = byte >> (7 - self.pos % 8) & 1;
        self.pos += 1;
        Ok(bit == 1)
    }

    /// `u(n)`, `n` at most 32.
    fn bits(&mut self, n: u32) -> Result<u32, String> {
        let mut value = 0;
        for _ in 0..n {
            value = value << 1 | u32::from(self.flag()?);
        }
        Ok(value)
    }

    fn skip(&mut self, n: u64) -> Result<(), String> {
        let end = (self.pos as u64).saturating_add(n);
        if end > self.data.len() as u64 * 8 {
            return Err(TRUNCATED.into());
        }
        self.pos = end as usize;
        Ok(())
    }

    /// `ue(v)`.
    fn ue(&mut self) -> Result<u32, String> {
        let mut zeros = 0;
        while !self.flag()? {
            zeros += 1;
            if zeros == 32 {
                return Err("H.264 exp-Golomb code is longer than 32 bits".into());
            }
        }
        Ok(((1u64 << zeros) - 1 + u64::from(self.bits(zeros)?)) as u32)
    }

    /// `ue(v)` that must not exceed `max`.
    fn ue_max(&mut self, max: u32, name: &str) -> Result<u32, String> {
        let value = self.ue()?;
        if value > max {
            return Err(format!("H.264 {name} {value} is out of range"));
        }
        Ok(value)
    }

    /// `se(v)`.
    fn se(&mut self) -> Result<i32, String> {
        let k = i64::from(self.ue()?);
        Ok(if k % 2 == 1 { (k + 1) / 2 } else { -(k / 2) } as i32)
    }
}

/// The RBSP of `nal` and its `nal_unit_type` and `nal_ref_idc`.
fn rbsp(nal: &[u8], expected: &[u8], name: &str) -> Result<(Vec<u8>, u8, u8), String> {
    let header = *nal.first().ok_or(TRUNCATED)?;
    let nal_type = header & 0x1f;
    if !expected.contains(&nal_type) {
        return Err(format!("NAL unit type {nal_type} is not {name}"));
    }
    Ok((unescape(&nal[1..]), nal_type, header >> 5 & 3))
}

/// A sequence parameter set (§7.3.2.1.1).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Sps {
    pub(crate) id: u32,
    pub(crate) profile_idc: u8,
    pub(crate) constraint_set3: bool,
    pub(crate) level_idc: u8,
    pub(crate) chroma_format_idc: u32,
    pub(crate) separate_colour_plane: bool,
    pub(crate) frame_mbs_only: bool,
    pub(crate) log2_max_frame_num: u32,
    pub(crate) poc_type: u32,
    pub(crate) log2_max_poc_lsb: u32,
    pub(crate) delta_pic_order_always_zero: bool,
    pub(crate) offset_for_non_ref_pic: i32,
    pub(crate) offset_for_top_to_bottom_field: i32,
    pub(crate) offsets_for_ref_frame: Vec<i32>,
    pub(crate) max_num_ref_frames: u32,
    pub(crate) width_mbs: u32,
    pub(crate) height_map_units: u32,
    /// The decoded picture size in pixels, whole macroblocks.
    pub(crate) coded: (usize, usize),
    /// The picture after frame cropping.
    pub(crate) visible: Region,
    /// `None` when absent or malformed.
    pub(crate) vui: Option<Vui>,
}

/// The VUI fields of an SPS that HAL reads (Annex E.1.1).
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Vui {
    pub(crate) full_range: Option<bool>,
    pub(crate) colour: Option<ColourDescription>,
    pub(crate) max_num_reorder_frames: Option<u32>,
    pub(crate) max_dec_frame_buffering: Option<u32>,
}

/// `colour_primaries`, `transfer_characteristics` and `matrix_coefficients`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ColourDescription {
    pub(crate) primaries: u8,
    pub(crate) transfer: u8,
    pub(crate) matrix: u8,
}

impl Sps {
    pub(crate) fn parse(nal: &[u8]) -> Result<Sps, String> {
        let (rbsp, _, _) = rbsp(nal, &[NAL_SPS], "an SPS")?;
        let mut r = BitReader::new(&rbsp);
        let profile_idc = r.bits(8)? as u8;
        let constraint_set3 = r.bits(8)? & 0x10 != 0;
        let level_idc = r.bits(8)? as u8;
        let id = r.ue_max(31, "seq_parameter_set_id")?;
        let (mut chroma_format_idc, mut separate_colour_plane) = (1, false);
        if HIGH_PROFILES.contains(&profile_idc) {
            chroma_format_idc = r.ue_max(3, "chroma_format_idc")?;
            if chroma_format_idc == 3 {
                separate_colour_plane = r.flag()?;
            }
            r.ue_max(6, "bit_depth_luma_minus8")?;
            r.ue_max(6, "bit_depth_chroma_minus8")?;
            r.flag()?;
            if r.flag()? {
                let lists = if chroma_format_idc == 3 { 12 } else { 8 };
                for i in 0..lists {
                    if r.flag()? {
                        skip_scaling_list(&mut r, if i < 6 { 16 } else { 64 })?;
                    }
                }
            }
        }
        let log2_max_frame_num = r.ue_max(12, "log2_max_frame_num_minus4")? + 4;
        let poc_type = r.ue_max(2, "pic_order_cnt_type")?;
        let mut sps = Sps {
            id,
            profile_idc,
            constraint_set3,
            level_idc,
            chroma_format_idc,
            separate_colour_plane,
            frame_mbs_only: true,
            log2_max_frame_num,
            poc_type,
            log2_max_poc_lsb: 0,
            delta_pic_order_always_zero: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            offsets_for_ref_frame: Vec::new(),
            max_num_ref_frames: 0,
            width_mbs: 0,
            height_map_units: 0,
            coded: (0, 0),
            visible: Region::new(0, 0, 0, 0),
            vui: None,
        };
        if poc_type == 0 {
            sps.log2_max_poc_lsb = r.ue_max(12, "log2_max_pic_order_cnt_lsb_minus4")? + 4;
        } else if poc_type == 1 {
            sps.delta_pic_order_always_zero = r.flag()?;
            sps.offset_for_non_ref_pic = r.se()?;
            sps.offset_for_top_to_bottom_field = r.se()?;
            let cycle = r.ue_max(255, "num_ref_frames_in_pic_order_cnt_cycle")?;
            for _ in 0..cycle {
                sps.offsets_for_ref_frame.push(r.se()?);
            }
        }
        sps.max_num_ref_frames = r.ue()?;
        r.flag()?;
        sps.width_mbs = r.ue()? + 1;
        sps.height_map_units = r.ue()? + 1;
        sps.frame_mbs_only = r.flag()?;
        if !sps.frame_mbs_only {
            r.flag()?;
        }
        r.flag()?;
        let (width_mbs, height_mbs) = (u64::from(sps.width_mbs), u64::from(sps.height_mbs()));
        if width_mbs * height_mbs > MAX_FRAME_MBS {
            return Err(format!(
                "H.264 picture of {width_mbs}x{height_mbs} macroblocks exceeds every level"
            ));
        }
        let (width, height) = (width_mbs as usize * 16, height_mbs as usize * 16);
        sps.coded = (width, height);
        sps.visible = Region::new(0, 0, width, height);
        if r.flag()? {
            let [left, right, top, bottom] = [r.ue()?, r.ue()?, r.ue()?, r.ue()?].map(u64::from);
            let field = if sps.frame_mbs_only { 1 } else { 2 };
            let (unit_x, unit_y) = match sps.chroma_array_type() {
                1 => (2, 2 * field),
                2 => (2, field),
                _ => (1, field),
            };
            let (crop_w, crop_h) = ((left + right) * unit_x, (top + bottom) * unit_y);
            if crop_w >= width as u64 || crop_h >= height as u64 {
                return Err(format!(
                    "H.264 frame cropping of {crop_w}x{crop_h} leaves no {width}x{height} picture"
                ));
            }
            sps.visible = Region::new(
                (left * unit_x) as usize,
                (top * unit_y) as usize,
                width - crop_w as usize,
                height - crop_h as usize,
            );
        }
        if r.flag()? {
            sps.vui = match parse_vui(&mut r) {
                Ok(vui) => Some(vui),
                Err(e) => {
                    log::debug!("H.264 SPS {id}: the VUI does not parse and is ignored: {e}");
                    None
                }
            };
        }
        Ok(sps)
    }

    /// `ChromaArrayType` (§7.4.2.1.1).
    fn chroma_array_type(&self) -> u32 {
        if self.separate_colour_plane {
            0
        } else {
            self.chroma_format_idc
        }
    }

    /// `FrameHeightInMbs`.
    fn height_mbs(&self) -> u32 {
        self.height_map_units
            .saturating_mul(2 - u32::from(self.frame_mbs_only))
    }

    /// The most pictures that can precede a picture in decode order and
    /// follow it in output order: the VUI's `max_num_reorder_frames`, at most
    /// 16, when present; otherwise 0 when `pic_order_cnt_type` is 2; otherwise
    /// 0 for profile 44, 86, 100, 110, 122 or 244 with `constraint_set3_flag`
    /// (§E.2.1); otherwise [`max_dpb_frames`] for the level and frame size.
    pub(crate) fn reorder_depth(&self) -> usize {
        if let Some(n) = self.vui.as_ref().and_then(|v| v.max_num_reorder_frames) {
            return (n as usize).min(MAX_DPB_FRAMES);
        }
        let intra = self.constraint_set3 && INTRA_PROFILES.contains(&self.profile_idc);
        if self.poc_type == 2 || intra {
            return 0;
        }
        max_dpb_frames(
            self.level_idc,
            self.constraint_set3,
            self.width_mbs,
            self.height_mbs(),
        )
    }
}

fn skip_scaling_list(r: &mut BitReader, size: usize) -> Result<(), String> {
    let (mut last, mut next) = (8, 8);
    for _ in 0..size {
        if next != 0 {
            let delta = r.se()?;
            if !(-128..=127).contains(&delta) {
                return Err(format!("H.264 delta_scale {delta} is out of range"));
            }
            next = (last + delta + 256) % 256;
        }
        if next != 0 {
            last = next;
        }
    }
    Ok(())
}

fn parse_vui(r: &mut BitReader) -> Result<Vui, String> {
    const EXTENDED_SAR: u32 = 255;
    let mut vui = Vui::default();
    if r.flag()? && r.bits(8)? == EXTENDED_SAR {
        r.bits(32)?;
    }
    if r.flag()? {
        r.flag()?;
    }
    if r.flag()? {
        r.bits(3)?;
        vui.full_range = Some(r.flag()?);
        if r.flag()? {
            vui.colour = Some(ColourDescription {
                primaries: r.bits(8)? as u8,
                transfer: r.bits(8)? as u8,
                matrix: r.bits(8)? as u8,
            });
        }
    }
    if r.flag()? {
        r.ue()?;
        r.ue()?;
    }
    if r.flag()? {
        r.skip(65)?;
    }
    let nal_hrd = r.flag()?;
    if nal_hrd {
        skip_hrd(r)?;
    }
    let vcl_hrd = r.flag()?;
    if vcl_hrd {
        skip_hrd(r)?;
    }
    if nal_hrd || vcl_hrd {
        r.flag()?;
    }
    r.flag()?;
    if r.flag()? {
        r.flag()?;
        for _ in 0..4 {
            r.ue()?;
        }
        vui.max_num_reorder_frames = Some(r.ue()?);
        vui.max_dec_frame_buffering = Some(r.ue()?);
    }
    Ok(vui)
}

/// Skips `hrd_parameters()` (§E.1.2).
fn skip_hrd(r: &mut BitReader) -> Result<(), String> {
    let cpb_count = r.ue_max(31, "cpb_cnt_minus1")? + 1;
    r.skip(8)?;
    for _ in 0..cpb_count {
        r.ue()?;
        r.ue()?;
        r.flag()?;
    }
    r.skip(20)
}

/// The DPB size in frames, `MaxDpbFrames` (§A.3.1, Table A-1): `MaxDpbMbs`
/// of the level over the frame size in macroblocks, at most 16. Level 11
/// with `constraint_set3` is level 1b; an unknown level gives 16.
pub(crate) fn max_dpb_frames(
    level_idc: u8,
    constraint_set3: bool,
    width_mbs: u32,
    height_mbs: u32,
) -> usize {
    let max_dpb_mbs: u64 = match level_idc {
        9 | 10 => 396,
        11 if constraint_set3 => 396,
        11 => 900,
        12 | 13 | 20 => 2376,
        21 => 4752,
        22 | 30 => 8100,
        31 => 18000,
        32 => 20480,
        40 | 41 => 32768,
        42 => 34816,
        50 => 110_400,
        51 | 52 => 184_320,
        60..=62 => 696_320,
        _ => return MAX_DPB_FRAMES,
    };
    let frame_mbs = u64::from(width_mbs) * u64::from(height_mbs);
    if frame_mbs == 0 {
        return MAX_DPB_FRAMES;
    }
    (max_dpb_mbs / frame_mbs).min(MAX_DPB_FRAMES as u64) as usize
}

/// A picture parameter set (§7.3.2.2), through `redundant_pic_cnt_present_flag`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Pps {
    pub(crate) id: u32,
    pub(crate) sps_id: u32,
    pub(crate) bottom_field_pic_order_in_frame_present: bool,
    /// `num_slice_groups_minus1 + 1`.
    pub(crate) slice_groups: u32,
    /// `num_ref_idx_l0_default_active_minus1 + 1` and the same for list 1.
    pub(crate) num_ref_idx_default: (u32, u32),
    pub(crate) weighted_pred: bool,
    pub(crate) weighted_bipred_idc: u32,
    pub(crate) redundant_pic_cnt_present: bool,
}

impl Pps {
    pub(crate) fn parse(nal: &[u8]) -> Result<Pps, String> {
        let (rbsp, _, _) = rbsp(nal, &[NAL_PPS], "a PPS")?;
        let mut r = BitReader::new(&rbsp);
        let id = r.ue_max(255, "pic_parameter_set_id")?;
        let sps_id = r.ue_max(31, "seq_parameter_set_id")?;
        r.flag()?;
        let bottom_field_pic_order_in_frame_present = r.flag()?;
        let groups = r.ue_max(7, "num_slice_groups_minus1")? + 1;
        if groups > 1 {
            match r.ue_max(6, "slice_group_map_type")? {
                0 => {
                    for _ in 0..groups {
                        r.ue()?;
                    }
                }
                2 => {
                    for _ in 1..groups {
                        r.ue()?;
                        r.ue()?;
                    }
                }
                3..=5 => {
                    r.flag()?;
                    r.ue()?;
                }
                6 => {
                    let units = u64::from(r.ue()?) + 1;
                    r.skip(units * u64::from(32 - (groups - 1).leading_zeros()))?;
                }
                _ => {}
            }
        }
        let num_ref_idx_default = (
            r.ue_max(31, "num_ref_idx_l0_default_active_minus1")? + 1,
            r.ue_max(31, "num_ref_idx_l1_default_active_minus1")? + 1,
        );
        let weighted_pred = r.flag()?;
        let weighted_bipred_idc = r.bits(2)?;
        if weighted_bipred_idc == 3 {
            return Err("H.264 weighted_bipred_idc 3 is out of range".into());
        }
        r.se()?;
        r.se()?;
        r.se()?;
        r.skip(2)?;
        let redundant_pic_cnt_present = r.flag()?;
        Ok(Pps {
            id,
            sps_id,
            bottom_field_pic_order_in_frame_present,
            slice_groups: groups,
            num_ref_idx_default,
            weighted_pred,
            weighted_bipred_idc,
            redundant_pic_cnt_present,
        })
    }
}

/// The SPS and PPS of a stream by id, as most recently received.
#[derive(Debug, Default)]
pub(crate) struct ParameterSets {
    sps: HashMap<u32, (Vec<u8>, Sps)>,
    pps: HashMap<u32, (Vec<u8>, Pps)>,
}

impl ParameterSets {
    /// Stores `nal` when it is an SPS or PPS, replacing the set with its id.
    /// Returns whether the stored set changed. Other NAL unit types are
    /// ignored and return `false`.
    pub(crate) fn update(&mut self, nal: &[u8]) -> Result<bool, String> {
        fn store<T>(map: &mut HashMap<u32, (Vec<u8>, T)>, id: u32, nal: &[u8], set: T) -> bool {
            if map.get(&id).is_some_and(|(stored, _)| stored == nal) {
                return false;
            }
            map.insert(id, (nal.to_vec(), set));
            true
        }
        match nal.first().map(|header| header & 0x1f) {
            Some(NAL_SPS) => {
                let sps = Sps::parse(nal)?;
                Ok(store(&mut self.sps, sps.id, nal, sps))
            }
            Some(NAL_PPS) => {
                let pps = Pps::parse(nal)?;
                Ok(store(&mut self.pps, pps.id, nal, pps))
            }
            _ => Ok(false),
        }
    }

    pub(crate) fn sps(&self, id: u32) -> Option<&Sps> {
        self.sps.get(&id).map(|(_, sps)| sps)
    }

    pub(crate) fn pps(&self, id: u32) -> Option<&Pps> {
        self.pps.get(&id).map(|(_, pps)| pps)
    }

    /// The SPS that PPS `pps_id` refers to.
    pub(crate) fn sps_for_pps(&self, pps_id: u32) -> Option<&Sps> {
        self.sps(self.pps(pps_id)?.sps_id)
    }

    /// The SPS and PPS NAL units, as received, of PPS `pps_id` and the SPS
    /// it refers to.
    pub(crate) fn nal_units_for_pps(&self, pps_id: u32) -> Option<(&[u8], &[u8])> {
        let (pps_nal, pps) = self.pps.get(&pps_id)?;
        let (sps_nal, _) = self.sps.get(&pps.sps_id)?;
        Some((sps_nal, pps_nal))
    }
}

/// A slice header (§7.3.3), through `dec_ref_pic_marking()`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SliceHeader {
    pub(crate) first_mb_in_slice: u32,
    /// As coded, 0 to 9.
    pub(crate) slice_type: u32,
    pub(crate) pps_id: u32,
    pub(crate) frame_num: u32,
    pub(crate) field_pic: bool,
    pub(crate) bottom_field: bool,
    pub(crate) idr: bool,
    pub(crate) nal_ref_idc: u8,
    pub(crate) poc_lsb: u32,
    pub(crate) delta_poc_bottom: i32,
    pub(crate) delta_poc: [i32; 2],
    /// `dec_ref_pic_marking()` holds `memory_management_control_operation` 5.
    pub(crate) mmco5: bool,
}

impl SliceHeader {
    /// Parses the header of a coded slice NAL unit against the parameter
    /// sets it refers to.
    pub(crate) fn parse(nal: &[u8], sets: &ParameterSets) -> Result<SliceHeader, String> {
        let (rbsp, nal_type, nal_ref_idc) = rbsp(nal, &[NAL_SLICE, NAL_IDR], "a coded slice")?;
        let mut r = BitReader::new(&rbsp);
        let idr = nal_type == NAL_IDR;
        let first_mb_in_slice = r.ue()?;
        let slice_type = r.ue_max(9, "slice_type")?;
        let pps_id = r.ue_max(255, "pic_parameter_set_id")?;
        let pps = sets
            .pps(pps_id)
            .ok_or_else(|| format!("H.264 slice refers to PPS {pps_id}, which is absent"))?;
        let sps = sets.sps(pps.sps_id).ok_or_else(|| {
            format!(
                "H.264 PPS {pps_id} refers to SPS {}, which is absent",
                pps.sps_id
            )
        })?;
        if sps.separate_colour_plane {
            r.skip(2)?;
        }
        let frame_num = r.bits(sps.log2_max_frame_num)?;
        let (mut field_pic, mut bottom_field) = (false, false);
        if !sps.frame_mbs_only {
            field_pic = r.flag()?;
            if field_pic {
                bottom_field = r.flag()?;
            }
        }
        if idr {
            r.ue()?;
        }
        let bottom_present = pps.bottom_field_pic_order_in_frame_present && !field_pic;
        let (mut poc_lsb, mut delta_poc_bottom, mut delta_poc) = (0, 0, [0; 2]);
        if sps.poc_type == 0 {
            poc_lsb = r.bits(sps.log2_max_poc_lsb)?;
            if bottom_present {
                delta_poc_bottom = r.se()?;
            }
        }
        if sps.poc_type == 1 && !sps.delta_pic_order_always_zero {
            delta_poc[0] = r.se()?;
            if bottom_present {
                delta_poc[1] = r.se()?;
            }
        }
        if pps.redundant_pic_cnt_present {
            r.ue()?;
        }
        let kind = slice_type % 5;
        let b = kind == SLICE_B;
        let p = kind == SLICE_P || kind == SLICE_SP;
        if b {
            r.flag()?;
        }
        let (mut l0, mut l1) = pps.num_ref_idx_default;
        if p || b {
            if r.flag()? {
                l0 = r.ue_max(31, "num_ref_idx_l0_active_minus1")? + 1;
                if b {
                    l1 = r.ue_max(31, "num_ref_idx_l1_active_minus1")? + 1;
                }
            }
            skip_ref_pic_list_modification(&mut r)?;
            if b {
                skip_ref_pic_list_modification(&mut r)?;
            }
        }
        if (pps.weighted_pred && p) || (pps.weighted_bipred_idc == 1 && b) {
            let lists = if b { [l0, l1] } else { [l0, 0] };
            skip_pred_weight_table(&mut r, sps.chroma_array_type() != 0, lists)?;
        }
        let mmco5 = nal_ref_idc != 0 && dec_ref_pic_marking_has_mmco5(&mut r, idr)?;
        Ok(SliceHeader {
            first_mb_in_slice,
            slice_type,
            pps_id,
            frame_num,
            field_pic,
            bottom_field,
            idr,
            nal_ref_idc,
            poc_lsb,
            delta_poc_bottom,
            delta_poc,
            mmco5,
        })
    }
}

/// Skips one list of `ref_pic_list_modification()` (§7.3.3.1).
fn skip_ref_pic_list_modification(r: &mut BitReader) -> Result<(), String> {
    if r.flag()? {
        loop {
            match r.ue()? {
                3 => break,
                0..=2 => {
                    r.ue()?;
                }
                idc => {
                    return Err(format!(
                        "H.264 modification_of_pic_nums_idc {idc} is out of range"
                    ))
                }
            }
        }
    }
    Ok(())
}

/// Skips `pred_weight_table()` (§7.3.3.2) for `lists` reference indices.
fn skip_pred_weight_table(r: &mut BitReader, chroma: bool, lists: [u32; 2]) -> Result<(), String> {
    r.ue_max(7, "luma_log2_weight_denom")?;
    if chroma {
        r.ue_max(7, "chroma_log2_weight_denom")?;
    }
    for count in lists {
        for _ in 0..count {
            if r.flag()? {
                r.se()?;
                r.se()?;
            }
            if chroma && r.flag()? {
                for _ in 0..4 {
                    r.se()?;
                }
            }
        }
    }
    Ok(())
}

/// Reads `dec_ref_pic_marking()` (§7.3.3.3) and returns whether it holds
/// `memory_management_control_operation` 5.
fn dec_ref_pic_marking_has_mmco5(r: &mut BitReader, idr: bool) -> Result<bool, String> {
    if idr {
        r.skip(2)?;
        return Ok(false);
    }
    let mut mmco5 = false;
    if r.flag()? {
        loop {
            match r.ue()? {
                0 => break,
                1 | 2 | 4 | 6 => {
                    r.ue()?;
                }
                3 => {
                    r.ue()?;
                    r.ue()?;
                }
                5 => mmco5 = true,
                op => {
                    return Err(format!(
                        "H.264 memory_management_control_operation {op} is out of range"
                    ))
                }
            }
        }
    }
    Ok(mmco5)
}

/// Picture order count across the pictures of a stream (§8.2.1).
#[derive(Debug, Default)]
pub(crate) struct PocCounter {
    prev_poc_msb: i64,
    prev_poc_lsb: i64,
    prev_frame_num_offset: i64,
    prev_frame_num: i64,
}

impl PocCounter {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Forgets the previous pictures, as an IDR picture does.
    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    /// The picture order count of the picture `slice` starts, called once
    /// per picture in decode order: `PicOrderCnt()` of a frame or field
    /// (§8.2.1, types 0, 1 and 2). A picture with MMCO 5 has the count it
    /// takes after its marking, 0.
    pub(crate) fn picture_order(&mut self, slice: &SliceHeader, sps: &Sps) -> i32 {
        if slice.idr {
            self.reset();
        }
        let frame_num = i64::from(slice.frame_num);
        let frame_num_offset = if slice.idr {
            0
        } else if self.prev_frame_num > frame_num {
            self.prev_frame_num_offset
                .wrapping_add(1 << sps.log2_max_frame_num)
        } else {
            self.prev_frame_num_offset
        };
        let reference = slice.nal_ref_idc != 0;
        let (top, bottom) = match sps.poc_type {
            0 => {
                let max_lsb = 1i64 << sps.log2_max_poc_lsb;
                let (lsb, prev_lsb) = (i64::from(slice.poc_lsb), self.prev_poc_lsb);
                let msb = if lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2 {
                    self.prev_poc_msb.wrapping_add(max_lsb)
                } else if lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2 {
                    self.prev_poc_msb.wrapping_sub(max_lsb)
                } else {
                    self.prev_poc_msb
                };
                if reference {
                    (self.prev_poc_msb, self.prev_poc_lsb) = (msb, lsb);
                }
                let top = msb.wrapping_add(lsb);
                if slice.field_pic {
                    (top, top)
                } else {
                    (top, top.wrapping_add(slice.delta_poc_bottom.into()))
                }
            }
            1 => {
                let cycle = &sps.offsets_for_ref_frame;
                let mut abs_frame_num = if cycle.is_empty() {
                    0
                } else {
                    frame_num_offset.wrapping_add(frame_num)
                };
                if !reference && abs_frame_num > 0 {
                    abs_frame_num -= 1;
                }
                let mut expected = 0i64;
                if abs_frame_num > 0 {
                    let len = cycle.len() as i64;
                    let per_cycle: i64 = cycle.iter().map(|&o| i64::from(o)).sum();
                    let in_cycle = ((abs_frame_num - 1) % len) as usize;
                    expected = ((abs_frame_num - 1) / len).wrapping_mul(per_cycle);
                    for &offset in &cycle[..=in_cycle] {
                        expected = expected.wrapping_add(offset.into());
                    }
                }
                if !reference {
                    expected = expected.wrapping_add(sps.offset_for_non_ref_pic.into());
                }
                let top = expected.wrapping_add(slice.delta_poc[0].into());
                let to_bottom = i64::from(sps.offset_for_top_to_bottom_field);
                if !slice.field_pic {
                    let bottom = top.wrapping_add(to_bottom);
                    (top, bottom.wrapping_add(slice.delta_poc[1].into()))
                } else if slice.bottom_field {
                    let bottom = expected.wrapping_add(to_bottom);
                    (top, bottom.wrapping_add(slice.delta_poc[0].into()))
                } else {
                    (top, top)
                }
            }
            _ => {
                let count = if slice.idr {
                    0
                } else {
                    let doubled = frame_num_offset.wrapping_add(frame_num).wrapping_mul(2);
                    doubled.wrapping_sub(i64::from(!reference))
                };
                (count, count)
            }
        };
        self.prev_frame_num_offset = frame_num_offset;
        self.prev_frame_num = frame_num;
        let poc = match (slice.field_pic, slice.bottom_field) {
            (false, _) => top.min(bottom),
            (true, false) => top,
            (true, true) => bottom,
        };
        if slice.mmco5 {
            self.prev_frame_num_offset = 0;
            self.prev_frame_num = 0;
            self.prev_poc_msb = 0;
            self.prev_poc_lsb = if slice.bottom_field {
                0
            } else {
                top.wrapping_sub(poc)
            };
            return 0;
        }
        poc as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAIN_SPS: &[u8] = &[
        0x67, 0x4d, 0x40, 0x28, 0xec, 0xa0, 0x3c, 0x01, 0x13, 0xf2, 0xe0, 0x22, 0x00, 0x00, 0x03,
        0x00, 0x02, 0x00, 0x00, 0x03, 0x00, 0x78, 0x1e, 0x30, 0x63, 0x2c,
    ];
    const MAIN_PPS: &[u8] = &[0x68, 0xeb, 0xe1, 0x92, 0xc8];
    const MAIN_REORDER_DEPTH: usize = 2;
    const SMALL_SPS: &[u8] = &[
        0x67, 0x42, 0xc0, 0x1e, 0xd9, 0x00, 0xa0, 0x3d, 0xb0, 0x11, 0x00, 0x00, 0x03, 0x00, 0x01,
        0x00, 0x00, 0x03, 0x00, 0x3c, 0x0f, 0x16, 0x2e, 0x48,
    ];
    const LARGE_SPS: &[u8] = &[
        0x67, 0x42, 0xc0, 0x1f, 0xd9, 0x00, 0x50, 0x05, 0xbb, 0x01, 0x10, 0x00, 0x00, 0x03, 0x00,
        0x10, 0x00, 0x00, 0x03, 0x03, 0xc0, 0xf1, 0x83, 0x24, 0x80,
    ];
    const RESOLUTION_CHANGE_PPS: &[u8] = &[0x68, 0xcb, 0x81, 0x12, 0xc8];

    /// The first 64 bytes of each slice NAL unit of `h264_1080p_main_bframes.h264`, in decode order.
    const MAIN_SLICES: [&str; 30] = [
    "65888400df79c6c1e630229af5858af7e250f90bc4c3c0401f7db002c7c387eba038ebd95a7110f45d503b6feb180db70c93307b2c652d5672aa72d1d639d2df",
    "419a216c4dffdaccd8ff595e3ee93ae8356820b63636e990f8fa3e7c58b5fb5fe581da5362e6def0b148afd91c5eb60b7d7085b811d22037c4a45f423ea4b38e",
    "419a423c21932984df5e2ee3b8a35113a83b5cf00fa0e9a42ec827500942af0a6bbb0922a158ad8926999db0093b31c3b092051dd63d39de9a943da94cacb64e",
    "419a6349e10f2653026f6f68f51da54a3d825aaf6efc203c1638842abd7e3177f44a4ebff9ca227816713ea20a08000cb1361e839156a280e3d71417808f77d2",
    "419a8449e10f26530223ff6638bbffe01ff777146f31dd2dffb45d6e49faa414de6ebdf13001eb214489efe162f259aa5e5d3aa3d3dee317ef4cf3174842111a",
    "419aa549e10f2653026f62eed4147068e99a0d6110003ead003ed9a5ad8c88d998d5ca34ef66afa9eab7dc525409fc12e2116d18c51109d9bc542a8061ecb081",
    "419ac649e10f26530223ff7ad379f64cbff936af9aa839c61d67c1dadaaa09850657b82a7fdf6fd5a28f3b6a256f12aa676a4811fafb46c6d9c7f650e03b359c",
    "419ae949e10f26530223ff6dbd0073fffdeb47330875eb5dd7f8e65ab50098e9f84670185584d2c403318630a04d95fe7a3901f71d5c2983f826740fffa0a569",
    "419f0745113c29ffd2de93bffa3d3e4291639efc0bf05a985615d8330b3ffc86a4d358002579d8e57c13775a8aa97ffd1f213c5a52c31e08f7ec3a949ec1f867",
    "019f286a42dfd4bc8b87df548474ded002a993881449120e8a9f65afa4c3dadeb6ab8e61566c979e2087ff335d87e372e27b908af41477b8334b4cf710bfa788",
    "419b2c49a84168994c088f76a75a744707da7cd34530a8770d0885ae560d14c4e1863c4e2ba1ad6373bcf9bd0581b51f1e39197c35358b4e6f0fcf5babf9b24d",
    "419f4a45112c29ffd2de93bec1e1678c6c6b1115a2f66259873a021e9069e30f944b2b93e301e1ca201d6dc485eb3e4cf1bf43e7e8dff1ba0a4e1b80c6b3e3b6",
    "019f6b6a42dfd4bc8b87df548474ded002a99388142ed76d7fed556344960ae0cccf8328fcbe1a6f0cd63d3de8841daf8278b9c9171cbeb267562a0eaec9da98",
    "419b6e49a8416c994c144c2dff96542837d59591871125cc47398337b82d0e5e20798dfb4c008e2141a9e1e1f7bb03c4e1bfa8496ab6a61fc6a2513cbd90c402",
    "019f8d6a42dfce7ce57b145f8bc219dcb741d09bf4bff0d3b0291d85139750680e816db56143856934bfdefb844bdf19af17dd5f03ea82312641f9638b00b55c",
    "658882005f6e7d9a80b46b3c72ef1f0075855cb6ee1abdf54cad66f15b6803d6cb2228dfbd61f8e1372c6b50bd6cfbb686a3f3710848aa4b75d5afe6aba4d2e2",
    "419a216c447f66a78d49208ffc04476c12986c14bf5f3f3917484e403e2ece801e4d069b17a8e935c9e9e8bc9a01adc000117e2ebbd2d936b5486653ec782930",
    "419a423c21932984476698f9134d9e5528ca052ef431a7f29e4c2a5d030d77783193d2474d8fcdbe07125cfc3aff926402653befc549dded5d1f3cbb8eb22122",
    "419a6449e10f2653053c4776a8a34cff45b2ad853f5fdab239dfa87b06625ac282b68b9363f88738244e9c880e45b6ee40307c5336ecece714b04ccfc438049f",
    "019e836a42dfd1291bd5f4a3c07bdd0df35fe6c9645a48d5f537a9d0ec00c132be4b1d06c1757730b06e023a28831a750bd0fee001fe6dbc603eddeb1f3e05d1",
    "419a8749e10f2653022bff830a84678192c450eb8d95aeaa08a43e8bb61bed05542d9947d78b4591dadb40343262d3a400142ab8afb9fd295b874e6ae3149849",
    "419ea545113c29ffd2d9159a90dbf8231d2aa12af1933761efbae536bac6950550cbba1f7fa2b43826290c55b4f5ff0014c9ae732a732e7b9634ebe7f4eda3b8",
    "019ec66a42dfd12954388d4fd1d4f8934be5948bfad74aac30df25329268b5acb3e004592ee4040fd21e852096db2aa4784f8b3276cbbf8c08aeafcb16dfcb14",
    "419aca49a84168994c08ef8ec9399505568fef2babf0cd5d0004cf2fc7b152c1646f13623d68da3c7fa3a3e0e1ccd8134398c3d6f54ef62395028c26ebf11992",
    "419ee845112c29ffd2de52973d6d49fe73072e04f422a3b1ebc6249bed9d1fda2b6fbb8b7109e214832257b0969e02d65c27e079f08386b73916de6dbd548505",
    "019f096a42dfd1291cb95f4cfd1d4f8934be5948bfad74aada8e0961f37383e0d1b89d59c555c3483792bf6a67714416def9606382ffaa519072fd42d6123afe",
    "419b0c49a8416c994c144c21ff89ea8ef24e35bace20243c7163ee4cbd3c229c5d131d6303ee53b570bf6ea2cffaa584e2f5796e99dccd2bbdea88074eef136a",
    "019f2b6a42dfd4bccb058e8f49a496c4309f8d0328b440c658f0c681df392163ea106c37ef443cf6bb73aa463d39137f7706494c0925a5103bd757686cfe3f7d",
    "419b2e49e10a52653052c2df9659645185eefc431276c2cedf0dc00cdb380b5acfc7e000001281c416dde4c248cee8e51481a35f93000a417300a3cc6c670c1e",
    "019f4d6a42dfce7c76a22a91f937c8ede40fffc1fc0867888881ca1f0ad8c00f35f8d4ffdd4420a03c75dae804b9cc349bfae675f9ec711eacd97d348937a028",
    ];

    /// The first 64 bytes of each slice NAL unit of `h264_resolution_change.h264`, in decode
    /// order: ten at 640x480, then ten at 1280x720.
    const RESOLUTION_CHANGE_SLICES: [&str; 20] = [
    "65888415f1088ab83023917870001eff0025e938f2bed8c8a7f11f412f49c6c57d98830e75a477d6d9659ef53d0ae8cf51b0658bd3d029530ec46fa40f7614a8",
    "419a382bf9e100272b8448880060bada6194065d6c640029d1153fe56a487c670cde38089bf9985a6df9df5265988095edbcdc6efbc651c3897592a52fb2dc37",
    "419a540af57ae52ee08066097fd21c6d08179264d0c69233f188bad222263ae8a2e678d24fc9fba88041ab0118effffffcb6183a8ad45628060c05900a019405",
    "419a6067297f13c2e13360132e2d264026bd51e7f3ddd6fb110b0ac0515ecf66b88bf6c1afd2565e9f5c1d6ffef88c57fffffe57e1186cea4eace06edc0e0a6b",
    "419a806732b550f8ae19bc6078bd3c035cab5acf39d9dbcb8237a2df96258ede11a08f37fff082e3c180b20e02255c0f315ce687020e3ef42dd84119c4491562",
    "419aa077334ee836221279fbbc4dff91e58ef57846823ea0247a03080908402087b807785997cda4aa0e0a8219151c045e1e91cfd6278067b98aba19bb4be111",
    "419ac077293ebc7770bf808bf75ffd944f2dca0d4d848570bc21cebdfd8f82611b3731c75deff7ea3ed3fbbe1175821f843e0c7bf02063f009464589e1c2c3de",
    "419ae021cd78263606dd7866e29db705dfcc46c77c23e1c2834b41952d0601cab81c4d5ccd019b9a043c4c12ebc0830e0288a714c04422665c5f83c1b01349a8",
    "419b0025cc4f4d3fbc11921279fbbc4f0efef40a4e908611ff89f4b8107d61b04429c53854d841214b8816383900d884953d9b023dfffffead7292b9ad2f846f",
    "419b202dcbe6c65f7053ee70de5851e18d6287fd5bfbf7c23e1cc1efc1ef807011e0dbb9c498708ffdde10e9705f0dc9e4f406fe0789ae612f316057eb212661",
    "65888415f1088ab83023917870001eff0025e938f2bed8c8a7f23e825e938d857d98830e75a477d2662c8f7ae7a15d1bd46c14b17a79c52a61d3b3db02d730a5",
    "419a382bf9e1009ce508911001006b731b8061824fc480814454ff95a921f19c3378e0226fe66169b7e77d4996620257b6f371bbef197c3897592a52fa3d0904",
    "419a540af57f726e08066097fd21c6d08179264d0c69233f188bad222263ae8a2e678d24fd0b0545299ed8ff8294095fffffff13c12b70430d8208bc5c2a6c25",
    "419a6067297f13c2e13360132e2d264026bd51e6eb76b7d8885856028af67b35c45fb60d7e92b2f4f7e0eb7ff7f842815b7fffffe27825c23ddf94113dfc0a3e",
    "419a806732b25430280f17a601ae55ad679cecede7e08de8b7e58963b2fc2140a5fffffff13c12bf08d022effbfc0a38746204a2b2a6216dde2e1c0490048fee",
    "419aa077334ee836221279fbbc4d7f23cb1df05784684c5bf09c366074fc1d3f01bb832ae785db9ffc17fc0a10dc500e28070a46600b88058e2b820aa04fe1ac",
    "419ac077293ebc7770bf808bf75ffd944f2dca0d4d848570bc21cebdfd0043e244717cbec7c71f967b3da079db7bf60100f842816b7fffffe278266e071818a1",
    "419ae021ccac9ca609c53b6e0bbf988f65f842814bffffffe278267c0a20840a1dfff6083b5c0c30d8248a714c2a6c2090af892c70710098ffffffffffffffff",
    "419b0025ccd3f046494a410bc7773e77c15e11fffff5623af034fffffffffffffffe14491a7e4fd7fca701ab4ad2f82b10812f843bf08c3609a0e4fc1c9f01a0",
    "419b202dcbe6c65f7315c2378238817c63f5e83ab72fcef82c7e087efdf5c083e0510207ffffffffffffffc29e090b0c8f44bcce66bf2e745866c60b5d63fd77",
    ];

    /// `pic_order_cnt_lsb` of each slice of `MAIN_SLICES`.
    const MAIN_POCS: [i32; 30] = [
        0, 2, 4, 6, 8, 10, 12, 18, 14, 16, 24, 20, 22, 28, 26, //
        0, 2, 4, 8, 6, 14, 10, 12, 20, 16, 18, 24, 22, 28, 26,
    ];
    /// The decode index of each displayed picture of `h264_1080p_main_bframes.h264`.
    const MAIN_DISPLAY_ORDER: [usize; 30] = [
        0, 1, 2, 3, 4, 5, 6, 8, 9, 7, 11, 12, 10, 14, 13, //
        15, 16, 17, 19, 18, 21, 22, 20, 24, 25, 23, 27, 26, 29, 28,
    ];

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// Whether each slice starts a new output epoch (IDR or MMCO 5), and its
    /// picture order count.
    fn picture_orders(
        sets: &ParameterSets,
        counter: &mut PocCounter,
        slices: &[Vec<u8>],
    ) -> Vec<(bool, i32)> {
        slices
            .iter()
            .map(|nal| {
                let slice = SliceHeader::parse(nal, sets).unwrap();
                let sps = sets.sps_for_pps(slice.pps_id).unwrap();
                (slice.idr || slice.mmco5, counter.picture_order(&slice, sps))
            })
            .collect()
    }

    /// Decode indices sorted by output epoch, then picture order count.
    fn display_order(pocs: &[(bool, i32)]) -> Vec<usize> {
        let mut epoch = 0;
        let mut keyed: Vec<_> = pocs
            .iter()
            .enumerate()
            .map(|(i, &(new_epoch, poc))| {
                epoch += usize::from(new_epoch && i > 0);
                (epoch, poc, i)
            })
            .collect();
        keyed.sort();
        keyed.into_iter().map(|(_, _, i)| i).collect()
    }

    #[derive(Default)]
    struct BitWriter {
        bytes: Vec<u8>,
        len: usize,
    }

    impl BitWriter {
        fn bits(&mut self, n: u32, value: u64) -> &mut Self {
            for i in (0..n).rev() {
                if self.len.is_multiple_of(8) {
                    self.bytes.push(0);
                }
                let last = self.bytes.len() - 1;
                self.bytes[last] |= ((value >> i & 1) as u8) << (7 - self.len % 8);
                self.len += 1;
            }
            self
        }

        fn flag(&mut self, set: bool) -> &mut Self {
            self.bits(1, u64::from(set))
        }

        fn ue(&mut self, value: u32) -> &mut Self {
            let coded = u64::from(value) + 1;
            let len = 64 - coded.leading_zeros();
            self.bits(len - 1, 0).bits(len, coded)
        }

        fn se(&mut self, value: i32) -> &mut Self {
            let mapped = if value > 0 {
                2 * value.unsigned_abs() - 1
            } else {
                2 * value.unsigned_abs()
            };
            self.ue(mapped)
        }

        /// The NAL unit: the bits written, RBSP trailing bits, and emulation
        /// prevention bytes.
        fn nal(&mut self) -> Vec<u8> {
            self.flag(true);
            while !self.len.is_multiple_of(8) {
                self.flag(false);
            }
            let mut out = Vec::new();
            let mut zeros = 0;
            for &b in &self.bytes {
                if zeros >= 2 && b <= 3 {
                    out.push(3);
                    zeros = 0;
                }
                zeros = if b == 0 { zeros + 1 } else { 0 };
                out.push(b);
            }
            out
        }
    }

    /// A High profile SPS with scaling lists, a NAL HRD and a full VUI.
    fn high_profile_sps() -> Vec<u8> {
        let mut w = BitWriter::default();
        w.bits(8, 0x67).bits(8, 100).bits(8, 0).bits(8, 40).ue(0);
        w.ue(1).ue(0).ue(0).flag(false).flag(true);
        w.flag(true).se(-8);
        for _ in 1..6 {
            w.flag(false);
        }
        w.flag(true);
        for _ in 0..64 {
            w.se(1);
        }
        w.flag(false);
        w.ue(0).ue(2).ue(1).flag(false).ue(79).ue(44);
        w.flag(true).flag(true).flag(false);
        w.flag(true).flag(false).flag(false);
        w.flag(true).bits(3, 5).flag(true).flag(true);
        w.bits(8, 6).bits(8, 1).bits(8, 6);
        w.flag(false).flag(false);
        w.flag(true)
            .ue(0)
            .bits(8, 0)
            .ue(1000)
            .ue(2000)
            .flag(false)
            .bits(20, 0);
        w.flag(false).flag(false).flag(false);
        w.flag(true)
            .flag(true)
            .ue(0)
            .ue(0)
            .ue(16)
            .ue(16)
            .ue(1)
            .ue(2);
        w.nal()
    }

    /// A 640x480 Main profile SPS with `pic_order_cnt_type` 1 (cycle of
    /// offsets 2 and 4, `offset_for_non_ref_pic` -1) or 0 (`MaxPicOrderCntLsb`
    /// 16), and a 4-bit `frame_num`.
    fn main_profile_sps(id: u32, poc_type: u32) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.bits(8, 0x67)
            .bits(8, 77)
            .bits(8, 0)
            .bits(8, 30)
            .ue(id)
            .ue(0)
            .ue(poc_type);
        if poc_type == 0 {
            w.ue(0);
        } else {
            w.flag(false).se(-1).se(0).ue(2).se(2).se(4);
        }
        w.ue(2)
            .flag(false)
            .ue(39)
            .ue(29)
            .flag(true)
            .flag(true)
            .flag(false)
            .flag(false);
        w.nal()
    }

    fn pps(id: u32, sps_id: u32) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.bits(8, 0x68)
            .ue(id)
            .ue(sps_id)
            .flag(false)
            .flag(false)
            .ue(0)
            .ue(0)
            .ue(0);
        w.flag(false)
            .bits(2, 0)
            .se(0)
            .se(0)
            .se(0)
            .flag(true)
            .flag(false)
            .flag(false);
        w.nal()
    }

    /// A frame slice: `kind` 0 (P), 1 (B) or 2 (I), `poc` the
    /// `pic_order_cnt_lsb` or `delta_pic_order_cnt[0]` by the SPS's type.
    fn slice(
        sps: &Sps,
        pps_id: u32,
        ref_idc: u8,
        kind: u32,
        frame_num: u32,
        poc: i32,
        mmco5: bool,
    ) -> Vec<u8> {
        let idr = kind == 2 && frame_num == 0 && !mmco5;
        let mut w = BitWriter::default();
        w.bits(8, u64::from(ref_idc << 5 | if idr { 5 } else { 1 }));
        w.ue(0)
            .ue(kind)
            .ue(pps_id)
            .bits(sps.log2_max_frame_num, frame_num.into());
        if idr {
            w.ue(0);
        }
        if sps.poc_type == 0 {
            w.bits(sps.log2_max_poc_lsb, poc as u64);
        } else {
            w.se(poc);
        }
        if kind == 1 {
            w.flag(true);
        }
        if kind != 2 {
            w.flag(false).flag(false);
        }
        if kind == 1 {
            w.flag(false);
        }
        if ref_idc != 0 {
            if idr {
                w.flag(false).flag(false);
            } else if mmco5 {
                w.flag(true).ue(5).ue(0);
            } else {
                w.flag(false);
            }
        }
        w.nal()
    }

    #[test]
    fn parses_the_1080p_main_fixture_sps() {
        let sps = Sps::parse(MAIN_SPS).unwrap();
        assert_eq!(
            (sps.profile_idc, sps.coded, sps.frame_mbs_only),
            (77, (1920, 1088), true)
        );
        assert_eq!(sps.visible, Region::new(0, 0, 1920, 1080));
        assert_eq!(sps.reorder_depth(), MAIN_REORDER_DEPTH);
        assert_eq!(
            (sps.level_idc, sps.poc_type, sps.log2_max_poc_lsb),
            (40, 0, 6)
        );
        assert_eq!((sps.max_num_ref_frames, sps.log2_max_frame_num), (4, 4));
        let vui = sps.vui.unwrap();
        assert_eq!((vui.full_range, vui.colour), (None, None));
        assert_eq!(vui.max_dec_frame_buffering, Some(4));

        let pps = Pps::parse(MAIN_PPS).unwrap();
        assert_eq!(
            (pps.id, pps.sps_id, pps.num_ref_idx_default),
            (0, 0, (3, 1))
        );
        assert_eq!((pps.weighted_pred, pps.weighted_bipred_idc), (true, 2));
        assert!(!pps.bottom_field_pic_order_in_frame_present && !pps.redundant_pic_cnt_present);
    }

    #[test]
    fn parses_both_sizes_of_the_resolution_change_fixture() {
        let (a, b) = (
            Sps::parse(SMALL_SPS).unwrap(),
            Sps::parse(LARGE_SPS).unwrap(),
        );
        assert_eq!(
            (a.profile_idc, a.visible),
            (66, Region::new(0, 0, 640, 480))
        );
        assert_eq!(b.visible, Region::new(0, 0, 1280, 720));
        assert_eq!(a.reorder_depth(), 0);
        assert_eq!((a.level_idc, b.level_idc, a.poc_type), (30, 31, 2));
        assert_eq!((b.width_mbs, b.height_map_units), (80, 45));
    }

    #[test]
    fn the_b_frame_fixture_displays_in_presentation_order() {
        let mut sets = ParameterSets::default();
        assert!(sets.update(MAIN_SPS).unwrap() && sets.update(MAIN_PPS).unwrap());
        assert!(!sets.update(MAIN_SPS).unwrap());
        let slices: Vec<_> = MAIN_SLICES.iter().map(|s| hex(s)).collect();
        let headers: Vec<_> = slices
            .iter()
            .map(|s| SliceHeader::parse(s, &sets).unwrap())
            .collect();
        assert!(headers
            .iter()
            .all(|h| h.first_mb_in_slice == 0 && !h.mmco5 && !h.field_pic));
        assert_eq!(
            (
                headers[0].slice_type,
                headers[0].idr,
                headers[0].nal_ref_idc
            ),
            (7, true, 3)
        );
        assert_eq!(
            (
                headers[8].slice_type,
                headers[8].frame_num,
                headers[8].nal_ref_idc
            ),
            (6, 8, 2)
        );
        assert_eq!((headers[9].slice_type, headers[9].nal_ref_idc), (6, 0));

        let pocs = picture_orders(&sets, &mut PocCounter::new(), &slices);
        assert_eq!(pocs.iter().map(|p| p.1).collect::<Vec<_>>(), MAIN_POCS);
        assert_eq!(display_order(&pocs), MAIN_DISPLAY_ORDER);
    }

    #[test]
    fn baseline_pictures_display_in_decode_order() {
        let mut sets = ParameterSets::default();
        let mut counter = PocCounter::new();
        let mut pocs = Vec::new();
        let parts = [
            (SMALL_SPS, &RESOLUTION_CHANGE_SLICES[..10], true),
            (LARGE_SPS, &RESOLUTION_CHANGE_SLICES[10..], false),
        ];
        for (sps, slices, new_pps) in parts {
            assert!(sets.update(sps).unwrap());
            assert_eq!(sets.update(RESOLUTION_CHANGE_PPS).unwrap(), new_pps);
            let slices: Vec<_> = slices.iter().map(|s| hex(s)).collect();
            pocs.extend(picture_orders(&sets, &mut counter, &slices));
        }
        let expected: Vec<i32> = (0..20).map(|i| i % 10 * 2).collect();
        assert_eq!(pocs.iter().map(|p| p.1).collect::<Vec<_>>(), expected);
        assert_eq!(display_order(&pocs), (0..20).collect::<Vec<_>>());
        assert_eq!(sets.sps(0).unwrap().visible, Region::new(0, 0, 1280, 720));
    }

    #[test]
    fn emulation_prevention_bytes_are_removed() {
        assert_eq!(
            unescape(&[0x67, 0, 0, 3, 1, 0, 0, 3]),
            vec![0x67, 0, 0, 1, 0, 0]
        );
        assert_eq!(unescape(&[0, 0, 3, 0, 0, 3, 3]), vec![0, 0, 0, 0, 3]);
    }

    #[test]
    fn dpb_size_follows_the_level_and_frame_size() {
        assert_eq!(max_dpb_frames(40, false, 120, 68), 4);
        assert_eq!(max_dpb_frames(30, false, 40, 30), 6);
        assert_eq!(max_dpb_frames(51, false, 240, 135), 5);
        assert_eq!(max_dpb_frames(51, false, 20, 15), 16);
        assert_eq!(max_dpb_frames(11, true, 11, 9), 4);
        assert_eq!(max_dpb_frames(11, false, 11, 9), 9);
    }

    #[test]
    fn vui_colour_and_range_are_read() {
        let sps = Sps::parse(&high_profile_sps()).unwrap();
        assert_eq!(
            (sps.profile_idc, sps.chroma_format_idc, sps.poc_type),
            (100, 1, 2)
        );
        assert_eq!(sps.coded, (1280, 720));
        let vui = sps.vui.as_ref().unwrap();
        assert_eq!(vui.full_range, Some(true));
        assert_eq!(
            vui.colour,
            Some(ColourDescription {
                primaries: 6,
                transfer: 1,
                matrix: 6
            })
        );
        assert_eq!(
            (vui.max_num_reorder_frames, vui.max_dec_frame_buffering),
            (Some(1), Some(2))
        );
        assert_eq!(sps.reorder_depth(), 1);
    }

    #[test]
    fn poc_type_1_and_mmco5_follow_the_standard() {
        let mut sets = ParameterSets::default();
        for nal in [
            main_profile_sps(0, 1),
            pps(0, 0),
            main_profile_sps(1, 0),
            pps(1, 1),
        ] {
            assert!(sets.update(&nal).unwrap());
        }
        let type1 = sets.sps(0).unwrap().clone();
        assert_eq!(type1.offsets_for_ref_frame, [2, 4]);
        assert_eq!(
            (
                type1.offset_for_non_ref_pic,
                type1.offset_for_top_to_bottom_field
            ),
            (-1, 0)
        );
        assert!(!type1.delta_pic_order_always_zero);
        // (nal_ref_idc, kind, frame_num, delta_pic_order_cnt[0], mmco5, expected POC)
        let type1_pictures = [
            (3, 2, 0, 0, false, 0),  // IDR
            (2, 0, 1, 0, false, 2),  // absFrameNum 1: offsets[0]
            (0, 1, 2, 0, false, 1),  // absFrameNum 2 - 1 = 1: 2 + offset_for_non_ref_pic
            (2, 0, 2, 0, false, 6),  // absFrameNum 2: offsets[0] + offsets[1]
            (2, 0, 3, 1, false, 9),  // absFrameNum 3: one cycle (6) + offsets[0], + delta 1
            (2, 0, 4, 0, false, 12), // absFrameNum 4: two cycles
            (2, 0, 5, 0, true, 0),   // 14 before MMCO 5, 0 after
            (2, 0, 1, 0, false, 2),  // prevFrameNumOffset and prevFrameNum are 0
        ];
        let type0 = sets.sps(1).unwrap().clone();
        // (nal_ref_idc, kind, frame_num, pic_order_cnt_lsb, mmco5, expected POC)
        let type0_pictures = [
            (3, 2, 0, 0, false, 0),
            (2, 0, 1, 8, false, 8),
            (2, 0, 2, 14, false, 14),
            (2, 0, 3, 2, false, 18), // the LSB wraps: PicOrderCntMsb 16
            (0, 1, 4, 0, false, 16), // relative to the previous reference picture
            (2, 0, 4, 6, true, 0),   // 22 before MMCO 5, 0 after
            (2, 0, 1, 4, false, 4),  // prevPicOrderCntMsb and Lsb are 0
        ];
        for (sps, pps_id, pictures) in
            [(type1, 0, &type1_pictures[..]), (type0, 1, &type0_pictures)]
        {
            let mut counter = PocCounter::new();
            for &(ref_idc, kind, frame_num, poc, mmco5, expected) in pictures {
                let nal = slice(&sps, pps_id, ref_idc, kind, frame_num, poc, mmco5);
                let header = SliceHeader::parse(&nal, &sets).unwrap();
                assert_eq!((header.mmco5, header.frame_num), (mmco5, frame_num));
                assert_eq!(counter.picture_order(&header, &sps), expected, "{header:?}");
            }
        }
    }

    /// A 640x480 (40x30 macroblock) SPS, with a VUI holding only
    /// `max_num_reorder_frames` when `reorder` is given.
    fn reorder_sps(profile: u8, set3: bool, level: u8, poc_type: u32, reorder: Option<u32>) -> Sps {
        let mut w = BitWriter::default();
        w.bits(8, 0x67)
            .bits(8, profile.into())
            .bits(8, if set3 { 0x10 } else { 0 });
        w.bits(8, level.into()).ue(0);
        if HIGH_PROFILES.contains(&profile) {
            w.ue(1).ue(0).ue(0).flag(false).flag(false);
        }
        w.ue(0).ue(poc_type);
        if poc_type == 0 {
            w.ue(0);
        }
        w.ue(2)
            .flag(false)
            .ue(39)
            .ue(29)
            .flag(true)
            .flag(true)
            .flag(false);
        w.flag(reorder.is_some());
        if let Some(n) = reorder {
            for _ in 0..8 {
                w.flag(false);
            }
            w.flag(true)
                .flag(true)
                .ue(0)
                .ue(0)
                .ue(16)
                .ue(16)
                .ue(n)
                .ue(4);
        }
        Sps::parse(&w.nal()).unwrap()
    }

    #[test]
    fn reorder_depth_falls_back_per_the_standard() {
        assert_eq!(reorder_sps(66, false, 30, 0, Some(2)).reorder_depth(), 2);
        assert_eq!(reorder_sps(77, false, 30, 2, Some(2)).reorder_depth(), 2);
        assert_eq!(reorder_sps(77, false, 30, 2, None).reorder_depth(), 0);
        assert_eq!(reorder_sps(66, false, 30, 2, None).reorder_depth(), 0);
        assert_eq!(reorder_sps(100, true, 30, 0, None).reorder_depth(), 0);
        // Table A-1 MaxDpbMbs: level 2.1 is 4752, 4752 / 1200 = 3.96.
        assert_eq!(reorder_sps(66, false, 21, 0, None).reorder_depth(), 3);
        // Level 3 is 8100, 8100 / 1200 = 6.75.
        assert_eq!(reorder_sps(66, false, 30, 0, None).reorder_depth(), 6);
        assert_eq!(reorder_sps(77, false, 30, 0, None).reorder_depth(), 6);
        // High profile without constraint_set3 is not intra-only.
        assert_eq!(reorder_sps(100, false, 30, 0, None).reorder_depth(), 6);
    }

    #[test]
    fn malformed_headers_are_errors() {
        assert!(Sps::parse(&[0x67]).is_err());
        assert!(Sps::parse(&[0x67, 100, 0, 40]).is_err());
        assert!(Sps::parse(&[]).is_err());
        assert!(Sps::parse(MAIN_PPS).is_err());
        assert!(Sps::parse(&[0x67, 77, 0, 40, 0, 0, 0, 0, 0x80]).is_err());
        assert!(Pps::parse(&[0x68]).is_err());
        assert!(Pps::parse(MAIN_SPS).is_err());
        let sets = ParameterSets::default();
        assert!(SliceHeader::parse(&hex(MAIN_SLICES[0]), &sets).is_err());
        assert!(SliceHeader::parse(MAIN_SPS, &sets).is_err());
    }

    #[test]
    fn truncated_headers_are_errors_not_panics() {
        for n in 0..=10 {
            assert!(Sps::parse(&MAIN_SPS[..n]).is_err(), "{n} bytes");
        }
        assert_eq!(Sps::parse(&MAIN_SPS[..11]).unwrap().vui, None);
        for n in 0..MAIN_PPS.len() {
            assert!(Pps::parse(&MAIN_PPS[..n]).is_err(), "{n} bytes");
        }
        let fixtures = [
            (&[MAIN_SPS, MAIN_PPS][..], &MAIN_SLICES[..]),
            (
                &[SMALL_SPS, RESOLUTION_CHANGE_PPS],
                &RESOLUTION_CHANGE_SLICES[..10],
            ),
            (
                &[LARGE_SPS, RESOLUTION_CHANGE_PPS],
                &RESOLUTION_CHANGE_SLICES[10..],
            ),
        ];
        for (parameter_sets, slices) in fixtures {
            let mut sets = ParameterSets::default();
            for nal in parameter_sets {
                for n in 0..nal.len() {
                    let _ = sets.update(&nal[..n]);
                }
                sets.update(nal).unwrap();
            }
            for slice in slices.iter().map(|s| hex(s)) {
                for n in 0..slice.len() {
                    let _ = SliceHeader::parse(&slice[..n], &sets);
                }
                SliceHeader::parse(&slice, &sets).unwrap();
            }
        }
    }

    #[test]
    fn corrupted_headers_never_panic() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as usize
        };
        let slices: Vec<_> = MAIN_SLICES.iter().map(|s| hex(s)).collect();
        let mut originals = vec![
            MAIN_SPS.to_vec(),
            MAIN_PPS.to_vec(),
            high_profile_sps(),
            main_profile_sps(0, 1),
        ];
        originals.extend(slices.iter().cloned());
        for _ in 0..20_000 {
            let mut nal = originals[next() % originals.len()].clone();
            for _ in 0..=next() % 4 {
                let at = next() % nal.len();
                nal[at] ^= 1 << (next() % 8);
            }
            let mut sets = ParameterSets::default();
            for set in [MAIN_SPS, MAIN_PPS, &nal] {
                let _ = sets.update(set);
            }
            let mut counter = PocCounter::new();
            for slice in std::iter::once(&nal).chain(&slices[..3]) {
                let Ok(header) = SliceHeader::parse(slice, &sets) else {
                    continue;
                };
                let sps = sets.sps_for_pps(header.pps_id).unwrap();
                counter.picture_order(&header, sps);
                sps.reorder_depth();
            }
        }
    }
}
