// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! SAHI-style input tiling: grid, plan, and GPU blit.

use std::ffi::{c_char, c_int, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_decoder_abi::{EfTilePlacement, TilePlacement};
use edgefirst_image::{tile_grid, Fit, TileSpec, TilingConfig};
use edgefirst_tensor::{DType, PixelFormat, TensorMemory};
use edgefirst_tensor_ffi::EfTensor;

use crate::processor::{cpu_access_from_code, with_tensor, with_tensor_mut, EfImageProcessor};

/// Static tiling configuration. Seed with [`ef_tiling_config_default`].
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfTilingConfig {
    pub tile_w: usize,
    pub tile_h: usize,
    pub overlap_ratio: f32,
    pub pad: [u8; 4],
    /// 0 = stretch, 1 = letterbox.
    pub fit: c_int,
}

impl From<EfTilingConfig> for TilingConfig {
    fn from(c: EfTilingConfig) -> Self {
        TilingConfig {
            tile_w: c.tile_w,
            tile_h: c.tile_h,
            overlap_ratio: c.overlap_ratio,
            pad: c.pad,
            fit: if c.fit != 0 {
                Fit::Letterbox { pad: c.pad }
            } else {
                Fit::Stretch
            },
        }
    }
}

/// One tile's native-frame crop and grid coordinates.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfTileSpec {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub index: usize,
    pub row: usize,
    pub col: usize,
}

impl From<&TileSpec> for EfTileSpec {
    fn from(t: &TileSpec) -> Self {
        Self {
            x: t.source.x as u32,
            y: t.source.y as u32,
            width: t.source.width as u32,
            height: t.source.height as u32,
            index: t.index,
            row: t.row,
            col: t.col,
        }
    }
}

/// Opaque list of tile specs.
pub struct EfTileSpecList {
    specs: Vec<TileSpec>,
}

/// Opaque list of tile placements.
pub struct EfTilePlacementList {
    placements: Vec<TilePlacement>,
}

/// Deploy defaults: overlap 0.2, stretch, pad `[114,114,114,255]`.
#[no_mangle]
pub extern "C" fn ef_tiling_config_default(tile_w: usize, tile_h: usize) -> EfTilingConfig {
    let cfg = TilingConfig::new(tile_w, tile_h);
    EfTilingConfig {
        tile_w: cfg.tile_w,
        tile_h: cfg.tile_h,
        overlap_ratio: cfg.overlap_ratio,
        pad: cfg.pad,
        fit: 0,
    }
}

/// EvenDist tile grid. Free with [`ef_tile_spec_list_free`].
#[no_mangle]
pub extern "C" fn ef_tile_grid(
    frame_h: usize,
    frame_w: usize,
    tile_h: usize,
    tile_w: usize,
    overlap_ratio: f32,
) -> *mut EfTileSpecList {
    catch_unwind(|| {
        if tile_w == 0 || tile_h == 0 || !(0.0..1.0).contains(&overlap_ratio) {
            return std::ptr::null_mut();
        }
        Box::into_raw(Box::new(EfTileSpecList {
            specs: tile_grid(frame_h, frame_w, tile_h, tile_w, overlap_ratio),
        }))
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Number of tile specs. Zero for NULL.
///
/// # Safety
/// `list` must be `NULL` or a live handle from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_tile_spec_list_len(list: *const EfTileSpecList) -> usize {
    unsafe {
        if list.is_null() {
            return 0;
        }
        catch_unwind(AssertUnwindSafe(|| (*list).specs.len())).unwrap_or(0)
    }
}

/// Copy one tile spec into `out`. Returns 0 on success.
///
/// # Safety
/// `list` and `out` must be live or NULL as documented.
#[no_mangle]
pub unsafe extern "C" fn ef_tile_spec_list_get(
    list: *const EfTileSpecList,
    index: usize,
    out: *mut EfTileSpec,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if list.is_null() || out.is_null() {
                return libc::EINVAL;
            }
            let specs = &(*list).specs;
            match specs.get(index) {
                Some(s) => {
                    *out = EfTileSpec::from(s);
                    0
                }
                None => libc::EINVAL,
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Free a tile-spec list. NULL is a no-op.
///
/// # Safety
/// `list` must be `NULL` or have come from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_tile_spec_list_free(list: *mut EfTileSpecList) {
    unsafe {
        if list.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(list))));
    }
}

/// Number of tile placements. Zero for NULL.
///
/// # Safety
/// `list` must be `NULL` or a live handle from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_tile_placement_list_len(list: *const EfTilePlacementList) -> usize {
    unsafe {
        if list.is_null() {
            return 0;
        }
        catch_unwind(AssertUnwindSafe(|| (*list).placements.len())).unwrap_or(0)
    }
}

/// Copy one tile placement into `out`. Returns 0 on success.
///
/// # Safety
/// `list` and `out` must be live or NULL as documented.
#[no_mangle]
pub unsafe extern "C" fn ef_tile_placement_list_get(
    list: *const EfTilePlacementList,
    index: usize,
    out: *mut EfTilePlacement,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if list.is_null() || out.is_null() {
                return libc::EINVAL;
            }
            let placements = &(*list).placements;
            match placements.get(index) {
                Some(p) => {
                    *out = EfTilePlacement::from(p);
                    0
                }
                None => libc::EINVAL,
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Free a tile-placement list. NULL is a no-op.
///
/// # Safety
/// `list` must be `NULL` or have come from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_tile_placement_list_free(list: *mut EfTilePlacementList) {
    unsafe {
        if list.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(list))));
    }
}

fn image_err(e: &edgefirst_image::Error) -> c_int {
    use edgefirst_image::Error as E;
    match e {
        E::CropInvalid(_) | E::InvalidShape(_) | E::NotAnImage | E::AliasedBuffers(_) => {
            libc::EINVAL
        }
        E::UnsupportedFormat(_) | E::NotSupported(_) | E::NotImplemented(_) | E::NoConverter => {
            libc::ENOTSUP
        }
        E::Io(io) => io.raw_os_error().unwrap_or(libc::EIO),
        _ => libc::EIO,
    }
}

/// Allocate a tall packed batch that stacks `n` tiles.
///
/// # Safety
/// `format` is a NUL-terminated wire code.
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_alloc_tile_batch(
    p: *mut EfImageProcessor,
    n: usize,
    config: *const EfTilingConfig,
    format: *const c_char,
    dtype: u32,
    storage: u32,
    access: u32,
) -> *mut EfTensor {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() || config.is_null() || format.is_null() {
                return std::ptr::null_mut();
            }
            let Ok(s) = CStr::from_ptr(format).to_str() else {
                return std::ptr::null_mut();
            };
            let Some(fmt) = PixelFormat::from_str_code(s) else {
                return std::ptr::null_mut();
            };
            let Some(dt) = DType::from_code(dtype) else {
                return std::ptr::null_mut();
            };
            let Some(acc) = cpu_access_from_code(access) else {
                return std::ptr::null_mut();
            };
            let cfg: TilingConfig = (*config).into();
            let mem = TensorMemory::from_code(storage);
            match (*p).inner.alloc_tile_batch(n, &cfg, fmt, dt, mem, acc) {
                Ok(t) => t.into_raw(),
                Err(_) => std::ptr::null_mut(),
            }
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Plan tile placements for a frame. Free with [`ef_tile_placement_list_free`].
///
/// # Safety
/// `p` and `config` must be live or NULL as documented.
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_plan_tiles(
    p: *mut EfImageProcessor,
    src_w: usize,
    src_h: usize,
    config: *const EfTilingConfig,
) -> *mut EfTilePlacementList {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() || config.is_null() {
                return std::ptr::null_mut();
            }
            let cfg: TilingConfig = (*config).into();
            match (*p).inner.plan_tiles(src_w, src_h, &cfg) {
                Ok(placements) => Box::into_raw(Box::new(EfTilePlacementList { placements })),
                Err(_) => std::ptr::null_mut(),
            }
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Render every tile of `src` into `dst`.
///
/// # Safety
/// Pointers must be live or NULL as documented.
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_tile_into(
    p: *mut EfImageProcessor,
    src: *const EfTensor,
    dst: *mut EfTensor,
    config: *const EfTilingConfig,
) -> *mut EfTilePlacementList {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() || config.is_null() {
                return std::ptr::null_mut();
            }
            let cfg: TilingConfig = (*config).into();
            let result = with_tensor(src, |s| {
                with_tensor_mut(dst, |d| (*p).inner.tile_into(s, d, &cfg))
            });
            match result {
                Ok(Ok(Ok(placements))) => {
                    Box::into_raw(Box::new(EfTilePlacementList { placements }))
                }
                _ => std::ptr::null_mut(),
            }
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Render one planned tile of `src` into `dst`.
///
/// # Safety
/// Pointers must be live or NULL as documented.
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_tile_one(
    p: *mut EfImageProcessor,
    src: *const EfTensor,
    dst: *mut EfTensor,
    placement: *const EfTilePlacement,
    config: *const EfTilingConfig,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() || placement.is_null() || config.is_null() {
                return libc::EINVAL;
            }
            let cfg: TilingConfig = (*config).into();
            let place: TilePlacement = (&*placement).into();
            let result = with_tensor(src, |s| {
                with_tensor_mut(dst, |d| (*p).inner.tile_one(s, d, &place, &cfg))
            });
            match result {
                Ok(Ok(Ok(()))) => 0,
                Ok(Ok(Err(e))) => image_err(&e),
                Ok(Err(e)) | Err(e) => e,
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}
