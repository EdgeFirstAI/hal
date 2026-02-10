// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "linux")]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]

include!("./ffi.rs");

use four_char_code::{four_char_code, FourCharCode};
use nix::ioctl_write_ptr;
use std::{
    ffi::{c_char, CStr},
    fmt::Display,
    os::{
        fd::RawFd,
        raw::{c_ulong, c_void},
    },
    ptr::null_mut,
    rc::Rc,
};

/// 8 bit grayscale, full range
// pub const GREY: FourCharCode = four_char_code!("Y800");
pub const YUYV: FourCharCode = four_char_code!("YUYV");
pub const RGBA: FourCharCode = four_char_code!("RGBA");
pub const RGB: FourCharCode = four_char_code!("RGB ");
pub const NV12: FourCharCode = four_char_code!("NV12");

const G2D_2_3_0: Version = Version::new(6, 4, 11, 1049711);

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    IoError(std::io::Error),
    LibraryError(libloading::Error),
    InvalidFormat(String),
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::IoError(err)
    }
}

impl From<libloading::Error> for Error {
    fn from(err: libloading::Error) -> Self {
        Error::LibraryError(err)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct G2DFormat(g2d_format);

impl G2DFormat {
    /// Try to create a G2DFormat from a FourCharCode
    /// Supported formats are RGB, RGBA, YUYV, NV12
    pub fn try_from(fourcc: FourCharCode) -> Result<Self> {
        fourcc.try_into()
    }

    /// Get the underlying g2d_format
    pub fn format(&self) -> g2d_format {
        self.0
    }
}

impl TryFrom<FourCharCode> for G2DFormat {
    type Error = Error;

    fn try_from(format: FourCharCode) -> Result<Self, Self::Error> {
        match format {
            RGB => Ok(G2DFormat(g2d_format_G2D_RGB888)),
            RGBA => Ok(G2DFormat(g2d_format_G2D_RGBA8888)),
            YUYV => Ok(G2DFormat(g2d_format_G2D_YUYV)),
            NV12 => Ok(G2DFormat(g2d_format_G2D_NV12)),
            // GREY => Ok(G2DFormat(g2d_format_G2D_NV12)),
            _ => Err(Error::InvalidFormat(format.to_string())),
        }
    }
}

impl TryFrom<G2DFormat> for FourCharCode {
    type Error = Error;

    /// Try to convert a G2DFormat to a FourCharCode
    /// Supported formats are RGB, RGBA, YUYV, NV12
    fn try_from(format: G2DFormat) -> Result<Self, Self::Error> {
        match format.0 {
            g2d_format_G2D_RGB888 => Ok(RGB),
            g2d_format_G2D_RGBA8888 => Ok(RGBA),
            g2d_format_G2D_YUYV => Ok(YUYV),
            g2d_format_G2D_NV12 => Ok(NV12),
            _ => Err(Error::InvalidFormat(format!(
                "Unsupported G2D format: {format:?}"
            ))),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct G2DPhysical(c_ulong);

impl G2DPhysical {
    pub fn new(fd: RawFd) -> Result<Self> {
        let phys = dma_buf_phys(0);
        let err = unsafe { ioctl_dma_buf_phys(fd, &phys.0).unwrap_or(1) };
        if err != 0 {
            return Err(std::io::Error::last_os_error().into());
        }

        Ok(G2DPhysical(phys.0))
    }

    pub fn address(&self) -> c_ulong {
        self.0
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct dma_buf_phys(std::ffi::c_ulong);

const DMA_BUF_BASE: u8 = b'b';
const DMA_BUF_IOCTL_PHYS: u8 = 10;
ioctl_write_ptr!(
    ioctl_dma_buf_phys,
    DMA_BUF_BASE,
    DMA_BUF_IOCTL_PHYS,
    std::ffi::c_ulong
);

impl TryFrom<RawFd> for G2DPhysical {
    type Error = Error;

    fn try_from(fd: RawFd) -> Result<Self, Self::Error> {
        G2DPhysical::new(fd)
    }
}

impl From<u64> for G2DPhysical {
    fn from(buf: u64) -> Self {
        G2DPhysical(buf)
    }
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Default, Copy)]
/// G2D library version as reported by _G2D_VERSION symbol
pub struct Version {
    pub major: i64,
    pub minor: i64,
    pub patch: i64,
    pub num: i64,
}

impl Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}.{}.{}:{}",
            self.major, self.minor, self.patch, self.num
        )
    }
}

impl Version {
    const fn new(major: i64, minor: i64, patch: i64, num: i64) -> Self {
        Version {
            major,
            minor,
            patch,
            num,
        }
    }
}

fn guess_version(g2d: &g2d) -> Option<Version> {
    unsafe {
        let version = g2d
            .__library
            .get::<*const *const c_char>(b"_G2D_VERSION")
            .map_or(None, |v| Some(*v));

        if let Some(v) = version {
            // Seems like the char sequence is `\n\0$VERSION$6.4.3:398061:d3dac3f35d$\n\0`
            // So we need to shift the ptr by two
            let ptr = (*v).byte_offset(2);
            let s = CStr::from_ptr(ptr).to_string_lossy().to_string();
            log::debug!("G2D Version string is {s}");
            // s = "$VERSION$6.4.3:398061:d3dac3f35d$\n"
            let mut version = G2D_2_3_0;
            if let Some(s) = s.strip_prefix("$VERSION$") {
                let parts: Vec<_> = s.split(':').collect();
                let v: Vec<_> = parts[0].split('.').collect();
                version.major = v
                    .first()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(version.major);
                version.minor = v
                    .get(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(version.minor);
                version.patch = v
                    .get(2)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(version.patch);
                version.num = parts
                    .get(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(version.num);
            }

            Some(version)
        } else {
            None
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct G2DSurface {
    pub format: g2d_format,
    pub planes: [::std::os::raw::c_ulong; 3usize],
    pub left: ::std::os::raw::c_int,
    pub top: ::std::os::raw::c_int,
    pub right: ::std::os::raw::c_int,
    pub bottom: ::std::os::raw::c_int,
    #[doc = "< buffer stride, in Pixels"]
    pub stride: ::std::os::raw::c_int,
    #[doc = "< surface width, in Pixels"]
    pub width: ::std::os::raw::c_int,
    #[doc = "< surface height, in Pixels"]
    pub height: ::std::os::raw::c_int,
    #[doc = "< alpha blending parameters"]
    pub blendfunc: g2d_blend_func,
    #[doc = "< value is 0 ~ 255"]
    pub global_alpha: ::std::os::raw::c_int,
    pub clrcolor: ::std::os::raw::c_int,
    pub rot: g2d_rotation,
}

impl Default for G2DSurface {
    fn default() -> Self {
        G2DSurface {
            format: g2d_format_G2D_RGB888,
            planes: [0, 0, 0],
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
            stride: 0,
            width: 0,
            height: 0,
            blendfunc: g2d_blend_func_G2D_ZERO,
            global_alpha: 255,
            clrcolor: 0,
            rot: g2d_rotation_G2D_ROTATION_0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct G2DSurfaceLegacy {
    pub format: g2d_format,
    pub planes: [::std::os::raw::c_int; 3usize],
    pub left: ::std::os::raw::c_int,
    pub top: ::std::os::raw::c_int,
    pub right: ::std::os::raw::c_int,
    pub bottom: ::std::os::raw::c_int,
    #[doc = "< buffer stride, in Pixels"]
    pub stride: ::std::os::raw::c_int,
    #[doc = "< surface width, in Pixels"]
    pub width: ::std::os::raw::c_int,
    #[doc = "< surface height, in Pixels"]
    pub height: ::std::os::raw::c_int,
    #[doc = "< alpha blending parameters"]
    pub blendfunc: g2d_blend_func,
    #[doc = "< value is 0 ~ 255"]
    pub global_alpha: ::std::os::raw::c_int,
    pub clrcolor: ::std::os::raw::c_int,
    pub rot: g2d_rotation,
}

impl Default for G2DSurfaceLegacy {
    fn default() -> Self {
        G2DSurfaceLegacy {
            format: g2d_format_G2D_RGB888,
            planes: [0, 0, 0],
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
            stride: 0,
            width: 0,
            height: 0,
            blendfunc: g2d_blend_func_G2D_ZERO,
            global_alpha: 255,
            clrcolor: 0,
            rot: g2d_rotation_G2D_ROTATION_0,
        }
    }
}

impl From<&G2DSurface> for G2DSurfaceLegacy {
    fn from(surface: &G2DSurface) -> Self {
        G2DSurfaceLegacy {
            format: surface.format,
            planes: [
                surface.planes[0] as ::std::os::raw::c_int,
                surface.planes[1] as ::std::os::raw::c_int,
                surface.planes[2] as ::std::os::raw::c_int,
            ],
            left: surface.left,
            top: surface.top,
            right: surface.right,
            bottom: surface.bottom,
            stride: surface.stride,
            width: surface.width,
            height: surface.height,
            blendfunc: surface.blendfunc,
            global_alpha: surface.global_alpha,
            clrcolor: surface.clrcolor,
            rot: surface.rot,
        }
    }
}

#[derive(Debug)]
pub struct G2D {
    pub lib: Rc<g2d>,
    pub handle: *mut c_void,
    pub version: Version,
}

impl G2D {
    pub fn new<P>(path: P) -> Result<Self>
    where
        P: AsRef<::std::ffi::OsStr>,
    {
        let lib = unsafe { g2d::new(path)? };
        let mut handle: *mut c_void = null_mut();

        if unsafe { lib.g2d_open(&mut handle) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }

        let version = guess_version(&lib).unwrap_or(G2D_2_3_0);

        Ok(Self {
            lib: Rc::new(lib),
            version,
            handle,
        })
    }

    pub fn version(&self) -> Version {
        self.version
    }

    pub fn clear(&self, dst: &mut G2DSurface, color: [u8; 4]) -> Result<()> {
        dst.clrcolor = i32::from_le_bytes(color);
        let ret = if self.version >= G2D_2_3_0 {
            unsafe {
                self.lib
                    .g2d_clear(self.handle, dst as *const _ as *mut g2d_surface)
            }
        } else {
            let dst: G2DSurfaceLegacy = (dst as &G2DSurface).into();
            unsafe {
                self.lib
                    .g2d_clear(self.handle, &dst as *const _ as *mut g2d_surface)
            }
        };

        if ret != 0 {
            return Err(std::io::Error::last_os_error().into());
        }

        if unsafe { self.lib.g2d_finish(self.handle) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        dst.clrcolor = 0;

        Ok(())
    }

    pub fn blit(&self, src: &G2DSurface, dst: &G2DSurface) -> Result<()> {
        let ret = if self.version >= G2D_2_3_0 {
            unsafe {
                self.lib.g2d_blit(
                    self.handle,
                    src as *const _ as *mut g2d_surface,
                    dst as *const _ as *mut g2d_surface,
                )
            }
        } else {
            let src: G2DSurfaceLegacy = src.into();
            let dst: G2DSurfaceLegacy = dst.into();

            unsafe {
                self.lib.g2d_blit(
                    self.handle,
                    &src as *const _ as *mut g2d_surface,
                    &dst as *const _ as *mut g2d_surface,
                )
            }
        };

        if ret != 0 {
            return Err(std::io::Error::last_os_error().into());
        }

        if unsafe { self.lib.g2d_finish(self.handle) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }

        Ok(())
    }

    pub fn set_bt601_colorspace(&mut self) -> Result<()> {
        if unsafe {
            self.lib
                .g2d_enable(self.handle, g2d_cap_mode_G2D_YUV_BT_601)
        } != 0
        {
            return Err(std::io::Error::last_os_error().into());
        }
        if unsafe {
            self.lib
                .g2d_disable(self.handle, g2d_cap_mode_G2D_YUV_BT_709)
        } != 0
        {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(())
    }

    pub fn set_bt709_colorspace(&mut self) -> Result<()> {
        if unsafe {
            self.lib
                .g2d_disable(self.handle, g2d_cap_mode_G2D_YUV_BT_601)
        } != 0
        {
            return Err(std::io::Error::last_os_error().into());
        }

        if unsafe {
            self.lib
                .g2d_disable(self.handle, g2d_cap_mode_G2D_YUV_BT_601FR)
        } != 0
        {
            return Err(std::io::Error::last_os_error().into());
        }

        if unsafe {
            self.lib
                .g2d_disable(self.handle, g2d_cap_mode_G2D_YUV_BT_709FR)
        } != 0
        {
            return Err(std::io::Error::last_os_error().into());
        }

        if unsafe {
            self.lib
                .g2d_enable(self.handle, g2d_cap_mode_G2D_YUV_BT_709)
        } != 0
        {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(())
    }
}

impl Drop for G2D {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                self.lib.g2d_close(self.handle);
            }
        }
    }
}
