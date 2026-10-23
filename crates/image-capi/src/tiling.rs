// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! SAHI-style input tiling: grid, plan, and GPU blit.

use std::ffi::{c_char, c_int, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_decoder_abi::{EfTilePlacement, TilePlacement};
use edgefirst_image::{tile_grid, TileSpec, TilingConfig};
use edgefirst_tensor::{DType, PixelFormat, TensorMemory};
use edgefirst_tensor_ffi::EfTensor;

use crate::processor::{
    cpu_access_from_code, image_errno, options_from, with_tensor, with_tensor_mut,
    EfConvertOptions, EfImageProcessor,
};

/// Static tiling geometry. Seed with [`ef_tiling_config_default`].
///
/// How each tile is converted (fit, letterbox placement and pad,
/// normalization) is the `const ef_convert_options *` the tiling functions
/// take beside it; `NULL` stretches each tile with the default mapping.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfTilingConfig {
    pub tile_w: usize,
    pub tile_h: usize,
    pub overlap_ratio: f32,
}

/// The Rust config: `c`'s geometry converted with `o`, or the default
/// options for `NULL`. `Err` carries the errno for an invalid config.
///
/// # Safety
/// `o` must be `NULL` or a live handle.
unsafe fn tiling_config_from(
    c: &EfTilingConfig,
    o: *const EfConvertOptions,
) -> Result<TilingConfig, c_int> {
    let mut cfg = TilingConfig::new(c.tile_w, c.tile_h).with_overlap(c.overlap_ratio);
    // SAFETY: the caller's contract.
    if let Some(options) = unsafe { options_from(o) } {
        cfg = cfg.with_options(options);
    }
    cfg.validate().map_err(|e| image_errno(&e))?;
    Ok(cfg)
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

/// Deploy defaults: tile size = model input, overlap 0.2.
#[no_mangle]
pub extern "C" fn ef_tiling_config_default(tile_w: usize, tile_h: usize) -> EfTilingConfig {
    let cfg = TilingConfig::new(tile_w, tile_h);
    EfTilingConfig {
        tile_w: cfg.tile_w,
        tile_h: cfg.tile_h,
        overlap_ratio: cfg.overlap_ratio,
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

/// Allocate a tall packed batch that stacks `n` tiles.
///
/// The declared access is widened as for `ef_image_processor_create_image`;
/// tiles are written through views, so any predicted CPU write is added as
/// `EF_CPU_ACCESS_READ_WRITE`.
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
            let Ok(cfg) = tiling_config_from(&*config, std::ptr::null()) else {
                return std::ptr::null_mut();
            };
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
/// `options` (`NULL` for a plain stretch) are the convert options every tile
/// is rendered with; a letterboxing one gives each placement a content
/// region. Pass the same options to [`ef_image_processor_tile_into`] or
/// [`ef_image_processor_tile_one`].
///
/// @return the list, or `NULL` with `errno` `EINVAL` for a `NULL` argument,
/// a zero tile size, an overlap outside `[0, 1)`, or options that rotate or
/// flip.
///
/// # Safety
/// `p` and `config` must be live or NULL as documented; `options` `NULL` or
/// live.
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_plan_tiles(
    p: *mut EfImageProcessor,
    src_w: usize,
    src_h: usize,
    config: *const EfTilingConfig,
    options: *const EfConvertOptions,
) -> *mut EfTilePlacementList {
    let fail = |code: c_int| {
        errno::set_errno(errno::Errno(code));
        std::ptr::null_mut()
    };
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() || config.is_null() {
                return fail(libc::EINVAL);
            }
            let cfg = match tiling_config_from(&*config, options) {
                Ok(cfg) => cfg,
                Err(code) => return fail(code),
            };
            match (*p).inner.plan_tiles(src_w, src_h, &cfg) {
                Ok(placements) => Box::into_raw(Box::new(EfTilePlacementList { placements })),
                Err(e) => fail(image_errno(&e)),
            }
        }))
        .unwrap_or_else(|_| fail(libc::EINVAL))
    }
}

/// Render every tile of `src` into `dst`, each converted with `options`
/// (`NULL` for a plain stretch) sampling the tile's crop.
///
/// @return a placement list to free with [`ef_tile_placement_list_free`], or
/// `NULL` with `errno` set: `EINVAL` for bad arguments or options that rotate
/// or flip, `EACCES` when a tensor's declared CPU access does not cover the
/// CPU pass, otherwise the codes `ef_image_processor_convert` returns.
///
/// # Safety
/// Pointers must be live or NULL as documented.
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_tile_into(
    p: *mut EfImageProcessor,
    src: *const EfTensor,
    dst: *mut EfTensor,
    config: *const EfTilingConfig,
    options: *const EfConvertOptions,
) -> *mut EfTilePlacementList {
    let fail = |code: c_int| {
        errno::set_errno(errno::Errno(code));
        std::ptr::null_mut()
    };
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() || config.is_null() {
                return fail(libc::EINVAL);
            }
            let cfg = match tiling_config_from(&*config, options) {
                Ok(cfg) => cfg,
                Err(code) => return fail(code),
            };
            let result = with_tensor(src, |s| {
                with_tensor_mut(dst, |d| (*p).inner.tile_into(s, d, &cfg))
            });
            match result {
                Ok(Ok(Ok(placements))) => {
                    Box::into_raw(Box::new(EfTilePlacementList { placements }))
                }
                Ok(Ok(Err(e))) => fail(image_errno(&e)),
                Ok(Err(code)) | Err(code) => fail(code),
            }
        }))
        .unwrap_or_else(|_| fail(libc::EINVAL))
    }
}

/// Render one planned tile of `src` into `dst`, converted with `options`
/// (`NULL` for a plain stretch) sampling the tile's crop.
///
/// @return 0 on success, `EINVAL` for bad arguments or options that rotate
/// or flip, otherwise the codes `ef_image_processor_convert` returns
/// (`EACCES` for a refused CPU access).
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
    options: *const EfConvertOptions,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() || placement.is_null() || config.is_null() {
                return libc::EINVAL;
            }
            let cfg = match tiling_config_from(&*config, options) {
                Ok(cfg) => cfg,
                Err(code) => return code,
            };
            let place: TilePlacement = (&*placement).into();
            let result = with_tensor(src, |s| {
                with_tensor_mut(dst, |d| (*p).inner.tile_one(s, d, &place, &cfg))
            });
            match result {
                Ok(Ok(Ok(()))) => 0,
                Ok(Ok(Err(e))) => image_errno(&e),
                Ok(Err(e)) | Err(e) => e,
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::{
        ef_convert_options_free, ef_convert_options_new, ef_convert_options_set_letterbox,
        ef_convert_options_set_letterbox_placement, ef_convert_options_set_rotation,
        ef_image_processor_free, ef_image_processor_new_with_backend,
    };

    #[test]
    fn the_default_config_is_the_rust_default() {
        let c = ef_tiling_config_default(640, 480);
        let want = TilingConfig::new(640, 480);
        assert_eq!(
            (c.tile_w, c.tile_h, c.overlap_ratio),
            (640, 480, want.overlap_ratio)
        );
        assert_eq!(std::mem::size_of::<EfTilingConfig>(), 24);
    }

    #[test]
    fn plan_tiles_carries_the_options_content_region() {
        let p = ef_image_processor_new_with_backend(1);
        assert!(!p.is_null(), "the CPU backend exists everywhere");
        let cfg = ef_tiling_config_default(640, 640);
        unsafe {
            let plain = ef_image_processor_plan_tiles(p, 1500, 400, &cfg, std::ptr::null());
            assert!(!plain.is_null());
            let mut place = std::mem::zeroed::<EfTilePlacement>();
            assert_eq!(ef_tile_placement_list_get(plain, 0, &mut place), 0);
            assert_eq!(place.has_content_region, 0, "a stretch fills the tile");
            ef_tile_placement_list_free(plain);

            let o = ef_convert_options_new();
            assert_eq!(
                ef_convert_options_set_letterbox(o, [114u8, 114, 114, 255].as_ptr()),
                0
            );
            assert_eq!(ef_convert_options_set_letterbox_placement(o, 1), 0);
            let boxed = ef_image_processor_plan_tiles(p, 1500, 400, &cfg, o);
            assert!(!boxed.is_null());
            let n = ef_tile_placement_list_len(boxed);
            assert!(n >= 3);
            for i in 0..n {
                assert_eq!(ef_tile_placement_list_get(boxed, i, &mut place), 0);
                assert_eq!(place.has_content_region, 1, "tile {i}");
                assert_eq!(place.content_region, [0.0, 0.0, 1.0, 0.625], "tile {i}");
            }
            ef_tile_placement_list_free(boxed);

            assert_eq!(ef_convert_options_set_rotation(o, 1), 0);
            errno::set_errno(errno::Errno(0));
            assert!(ef_image_processor_plan_tiles(p, 1500, 400, &cfg, o).is_null());
            assert_eq!(errno::errno().0, libc::EINVAL, "tiles cannot rotate");
            ef_convert_options_free(o);

            let bad = EfTilingConfig {
                overlap_ratio: 1.0,
                ..cfg
            };
            assert!(ef_image_processor_plan_tiles(p, 1500, 400, &bad, std::ptr::null()).is_null());
            ef_image_processor_free(p);
        }
    }
}
