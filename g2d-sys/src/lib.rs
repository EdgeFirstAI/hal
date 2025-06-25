#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]

include!("./ffi.rs");
mod ffi_new;
pub mod fourcc;
use dma_buf::DmaBuf;
pub use ffi_new::*;
use fourcc::FourCC;
use nix::ioctl_write_ptr;
use std::{
    ffi::{CStr, c_char},
    fmt::Display,
    os::fd::AsRawFd,
};

const RGB3: FourCC = FourCC(*b"RGB3");
const RGBX: FourCC = FourCC(*b"RGBX");
const RGBA: FourCC = FourCC(*b"RGBA");
const YUYV: FourCC = FourCC(*b"YUYV");
const NV12: FourCC = FourCC(*b"NV12");

pub struct G2DFormat(g2d_format);

impl G2DFormat {
    pub fn from(fourcc: FourCC) -> Self {
        fourcc.into()
    }

    pub fn format(&self) -> g2d_format {
        self.0
    }
}

impl From<FourCC> for G2DFormat {
    fn from(format: FourCC) -> Self {
        match format {
            RGB3 => G2DFormat(g2d_format_G2D_RGB888),
            RGBX => G2DFormat(g2d_format_G2D_RGBX8888),
            RGBA => G2DFormat(g2d_format_G2D_RGBA8888),
            YUYV => G2DFormat(g2d_format_G2D_YUYV),
            NV12 => G2DFormat(g2d_format_G2D_NV12),
            _ => todo!(),
        }
    }
}

impl From<G2DFormat> for FourCC {
    fn from(format: G2DFormat) -> Self {
        match format.0 {
            g2d_format_G2D_RGB888 => RGB3,
            g2d_format_G2D_RGBX8888 => RGBX,
            g2d_format_G2D_RGBA8888 => RGBA,
            g2d_format_G2D_YUYV => YUYV,
            g2d_format_G2D_NV12 => NV12,
            _ => todo!(),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct G2DPhysical(g2d_phys_addr_t_new);

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

impl From<DmaBuf> for G2DPhysical {
    fn from(buf: DmaBuf) -> Self {
        let phys = dma_buf_phys(0);
        let err = unsafe { ioctl_dma_buf_phys(buf.as_raw_fd(), &phys.0).unwrap_or(0) };
        if err != 0 {
            return G2DPhysical(0);
        }

        G2DPhysical(phys.0)
    }
}

impl From<u64> for G2DPhysical {
    fn from(buf: u64) -> Self {
        G2DPhysical(buf)
    }
}

impl From<G2DPhysical> for g2d_phys_addr_t {
    fn from(phys: G2DPhysical) -> Self {
        phys.0 as g2d_phys_addr_t
    }
}

impl From<G2DPhysical> for g2d_phys_addr_t_new {
    fn from(phys: G2DPhysical) -> g2d_phys_addr_t_new {
        phys.0
    }
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Default, Copy)]
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
    pub fn new(major: i64, minor: i64, patch: i64, num: i64) -> Self {
        Version {
            major,
            minor,
            patch,
            num,
        }
    }
}

pub fn guess_version(g2d: &g2d) -> Option<Version> {
    unsafe {
        let version = g2d
            .__library
            .get::<*const *const c_char>(b"_G2D_VERSION")
            .map_or(None, |v| Some(*v));

        // Seems like the char sequence is `\n\0$VERSION$6.4.3:398061:d3dac3f35d$\n\0`
        // So we need to shift the ptr by two
        let ptr = (*version.unwrap()).byte_offset(2);
        let s = CStr::from_ptr(ptr).to_string_lossy().to_string();
        // s = "$VERSION$6.4.3:398061:d3dac3f35d$\n"
        let mut version = Version::default();
        let s: Vec<_> = s[9..].split(":").collect();

        let v: Vec<_> = s[0].split(".").collect();
        if let Some(s) = v.first() {
            if let Ok(major) = s.parse() {
                version.major = major;
            }
        }
        if let Some(s) = v.get(1) {
            if let Ok(minor) = s.parse() {
                version.minor = minor;
            }
        }
        if let Some(s) = v.get(2) {
            if let Ok(patch) = s.parse() {
                version.patch = patch;
            }
        }
        if let Some(s) = s.get(1) {
            if let Ok(num) = s.parse() {
                version.num = num;
            }
        }
        Some(version)
    }
}
