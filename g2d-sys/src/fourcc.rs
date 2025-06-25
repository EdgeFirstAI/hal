// Taken from https://docs.rs/crate/four-cc/latest and adapted to handle endianess.
#![forbid(unsafe_code)]

use core::{fmt, result::Result};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C, packed)]
pub struct FourCC(pub [u8; 4]);

impl FourCC {
    const fn from_u32(self) -> u32 {
        #[cfg(target_endian = "little")]
        {
            ((self.0[3] as u32) << 24 & 0xff000000)
                | ((self.0[2] as u32) << 16 & 0x00ff0000)
                | ((self.0[1] as u32) << 8 & 0x0000ff00)
                | ((self.0[0] as u32) & 0x000000ff)
        }
        #[cfg(target_endian = "big")]
        {
            ((self.0[0] as u32) << 24 & 0xff000000)
                | ((self.0[1] as u32) << 16 & 0x00ff0000)
                | ((self.0[2] as u32) << 8 & 0x0000ff00)
                | ((self.0[3] as u32) & 0x000000ff)
        }
    }
}

impl<'a> From<&'a [u8; 4]> for FourCC {
    fn from(buf: &[u8; 4]) -> FourCC {
        FourCC([buf[0], buf[1], buf[2], buf[3]])
    }
}
impl<'a> From<&'a [u8]> for FourCC {
    fn from(buf: &[u8]) -> FourCC {
        FourCC([buf[0], buf[1], buf[2], buf[3]])
    }
}
impl From<u32> for FourCC {
    fn from(val: u32) -> FourCC {
        #[cfg(target_endian = "little")]
        {
            FourCC([
                (val & 0xff) as u8,
                (val >> 8 & 0xff) as u8,
                (val >> 16 & 0xff) as u8,
                (val >> 24 & 0xff) as u8,
            ])
        }
        #[cfg(target_endian = "big")]
        {
            FourCC([
                (val >> 24 & 0xff) as u8,
                (val >> 16 & 0xff) as u8,
                (val >> 8 & 0xff) as u8,
                (val & 0xff) as u8,
            ])
        }
    }
}

macro_rules! from_fourcc_for_u32 {
    ($($t:tt)*) => {
        impl From<FourCC> for u32 {
            fn from(val: FourCC) -> Self {
                val.from_u32()
            }
        }
    };
}
from_fourcc_for_u32!();

impl fmt::Display for FourCC {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match core::str::from_utf8(&self.0) {
            Ok(s) => f.write_str(s),
            Err(_) => {
                // If we return fmt::Error, then for example format!() will panic, so we choose
                // an alternative representation instead
                let b = &self.0;
                f.write_fmt(format_args!(
                    "{}{}{}{}",
                    core::ascii::escape_default(b[0]),
                    core::ascii::escape_default(b[1]),
                    core::ascii::escape_default(b[2]),
                    core::ascii::escape_default(b[3])
                ))
            }
        }
    }
}

impl fmt::Debug for FourCC {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let b = self.0;
        f.debug_tuple("FourCC")
            .field(&format_args!(
                "{}{}{}{}",
                core::ascii::escape_default(b[0]),
                core::ascii::escape_default(b[1]),
                core::ascii::escape_default(b[2]),
                core::ascii::escape_default(b[3])
            ))
            .finish()
    }
}
