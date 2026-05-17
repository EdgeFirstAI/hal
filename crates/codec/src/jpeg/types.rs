// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! JPEG data types: component descriptors, sampling factors, image header.

/// JPEG marker byte constants.
pub mod marker {
    pub const SOF0: u8 = 0xC0; // Baseline DCT (supported)
    pub const SOF1: u8 = 0xC1; // Extended sequential DCT
    pub const SOF2: u8 = 0xC2; // Progressive DCT
    pub const SOF3: u8 = 0xC3; // Lossless (sequential)
    pub const DHT: u8 = 0xC4; // Define Huffman Table
    pub const SOF5: u8 = 0xC5; // Differential sequential DCT
    pub const SOF6: u8 = 0xC6; // Differential progressive DCT
    pub const SOF7: u8 = 0xC7; // Differential lossless (sequential)
    pub const SOF9: u8 = 0xC9; // Extended sequential DCT, arithmetic
    pub const SOF10: u8 = 0xCA; // Progressive DCT, arithmetic
    pub const SOF11: u8 = 0xCB; // Lossless, arithmetic
    pub const SOF13: u8 = 0xCD; // Differential sequential DCT, arithmetic
    pub const SOF14: u8 = 0xCE; // Differential progressive DCT, arithmetic
    pub const SOF15: u8 = 0xCF; // Differential lossless, arithmetic
    pub const SOI: u8 = 0xD8; // Start of Image
    pub const EOI: u8 = 0xD9; // End of Image
    pub const SOS: u8 = 0xDA; // Start of Scan
    pub const DQT: u8 = 0xDB; // Define Quantisation Table
    pub const DRI: u8 = 0xDD; // Define Restart Interval
    pub const RST0: u8 = 0xD0; // Restart marker 0
    pub const RST7: u8 = 0xD7; // Restart marker 7
    pub const APP1: u8 = 0xE1; // Application segment 1 (EXIF)
}

/// Chroma sub-sampling factors for one component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplingFactor {
    /// Horizontal sampling factor (1–4).
    pub h: u8,
    /// Vertical sampling factor (1–4).
    pub v: u8,
}

/// Descriptor for one colour component (Y, Cb, or Cr).
#[derive(Debug, Clone, Copy)]
pub struct Component {
    /// Component identifier (1=Y, 2=Cb, 3=Cr for YCbCr).
    pub id: u8,
    /// Sub-sampling factors.
    pub sampling: SamplingFactor,
    /// Quantisation table index (0–3).
    pub quant_table_id: u8,
    /// DC Huffman table index assigned by SOS.
    pub dc_table_id: u8,
    /// AC Huffman table index assigned by SOS.
    pub ac_table_id: u8,
}

/// Parsed SOF (Start of Frame) header.
#[derive(Debug, Clone)]
pub struct ImageHeader {
    /// Image width in pixels.
    pub width: u16,
    /// Image height in pixels.
    pub height: u16,
    /// Component descriptors.
    pub components: Vec<Component>,
    /// Maximum horizontal sampling factor across all components.
    pub max_h_samp: u8,
    /// Maximum vertical sampling factor across all components.
    pub max_v_samp: u8,
    /// Whether this is a progressive scan (SOF2).
    pub is_progressive: bool,
}

impl ImageHeader {
    /// MCU width in pixels.
    pub fn mcu_width(&self) -> usize {
        self.max_h_samp as usize * 8
    }

    /// MCU height in pixels.
    pub fn mcu_height(&self) -> usize {
        self.max_v_samp as usize * 8
    }

    /// Number of MCU columns.
    pub fn mcus_x(&self) -> usize {
        (self.width as usize).div_ceil(self.mcu_width())
    }

    /// Number of MCU rows.
    pub fn mcus_y(&self) -> usize {
        (self.height as usize).div_ceil(self.mcu_height())
    }
}

/// 8×8 quantisation table stored in zig-zag order.
#[derive(Debug, Clone)]
pub struct QuantTable {
    /// 64 quantisation values in zig-zag scan order.
    pub values: [u16; 64],
}

impl Default for QuantTable {
    fn default() -> Self {
        Self { values: [1; 64] }
    }
}

/// Standard JPEG zig-zag scan order.
pub const ZIGZAG: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];
