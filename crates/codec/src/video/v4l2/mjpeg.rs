// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! MJPEG decoding on the V4L2 JPEG decoder (i.MX `mxc-jpeg`).
//!
//! Each picture is a JPEG of its own, decoded on the hardware by the image
//! path's V4L2 backend into a tensor of the pool, so there is no reordering
//! and nothing to drain. A picture the hardware cannot decode is an error:
//! MJPEG is a video codec, which HAL does not decode on the CPU.

use std::sync::Arc;

use edgefirst_tensor::{
    Colorimetry, CpuAccess, DType, PixelFormat, Region, TensorDyn, TensorMemory,
};

use crate::video::{
    DecodeEvent, DecodedFrame, DecoderBackend, DecoderConfig, OutputMemory, OutputPool, Returns,
    StreamInfo,
};
use crate::{peek_info, CodecError, Engine, ImageDecoder, Result, Transfer};

pub(crate) struct V4l2MjpegDecoder {
    decoder: ImageDecoder,
    device: String,
    output: OutputPool,
    pool: Vec<Arc<TensorDyn>>,
    free: Vec<u32>,
    /// Slots of the current pool given back by frames whose tensor reference
    /// has not dropped yet.
    releasing: Vec<u32>,
    generation: u64,
    returns: Arc<Returns>,
    info: Option<StreamInfo>,
}

impl V4l2MjpegDecoder {
    pub fn open(config: DecoderConfig) -> Result<Self> {
        let device = crate::jpeg::v4l2_device()
            .ok_or_else(|| CodecError::NoDevice("no V4L2 JPEG decoder".into()))?;
        let mut decoder = ImageDecoder::new();
        decoder.set_engine(Engine::V4l2);
        log::info!("MJPEG decoder {device}");
        Ok(Self {
            decoder,
            device,
            output: config.output,
            pool: Vec::new(),
            free: Vec::new(),
            releasing: Vec::new(),
            generation: 0,
            returns: Arc::new(Returns::default()),
            info: None,
        })
    }

    /// Gives back the buffers of frames the caller has dropped, once the
    /// pool holds each tensor's last reference.
    fn reclaim(&mut self) {
        let mut returned = std::mem::take(&mut self.releasing);
        returned.extend(self.returns.take(self.generation));
        for slot in returned {
            if Arc::strong_count(&self.pool[slot as usize]) == 1 {
                self.free.push(slot);
            } else {
                self.releasing.push(slot);
            }
        }
    }

    /// Makes the pool hold `w`x`h` pictures in `format` and describes them.
    ///
    /// # Errors
    ///
    /// [`CodecError::PoolExhausted`] when a provided pool must follow a new
    /// size while frames still hold some of its tensors; nothing changes.
    fn reconfigure(&mut self, w: usize, h: usize, format: PixelFormat) -> Result<StreamInfo> {
        let tensors = match &mut self.output {
            OutputPool::Allocate { count } => (0..*count + 1)
                .map(|_| {
                    TensorDyn::image(
                        w,
                        h,
                        format,
                        DType::U8,
                        Some(TensorMemory::DmaBuf),
                        CpuAccess::ReadWrite,
                    )
                })
                .collect::<std::result::Result<Vec<_>, _>>()?,
            // Provided tensors are kept across sizes: each decode configures
            // its tensor to the picture, within the tensor's capacity.
            OutputPool::Provided(pool) if !pool.is_empty() => std::mem::take(pool),
            OutputPool::Provided(_) => {
                let held = self
                    .pool
                    .iter()
                    .filter(|t| Arc::strong_count(t) > 1)
                    .count();
                if held > 0 {
                    return Err(CodecError::PoolExhausted {
                        held,
                        pool: self.pool.len(),
                    });
                }
                std::mem::take(&mut self.pool)
                    .into_iter()
                    .map(|t| Arc::try_unwrap(t).expect("no frame holds the tensor"))
                    .collect()
            }
        };
        self.generation += 1;
        self.pool = tensors.into_iter().map(Arc::new).collect();
        self.free = (0..self.pool.len() as u32).rev().collect();
        self.releasing.clear();
        Ok(StreamInfo {
            width: w,
            height: h,
            crop: Region::new(0, 0, w, h),
            coded_width: w,
            coded_height: h,
            format,
            colorimetry: Some(Colorimetry::jfif()),
            buffers: self.pool.len(),
            memory: OutputMemory::Imported,
            engine: Engine::V4l2,
            // Set from the first picture, which shows where the hardware
            // writes.
            transfer: Transfer::ZeroCopy,
        })
    }
}

impl DecoderBackend for V4l2MjpegDecoder {
    fn decode(&mut self, access_unit: &[u8], pts: u64) -> Result<Vec<DecodeEvent>> {
        if access_unit.is_empty() {
            return Err(CodecError::InvalidInput("empty access unit".into()));
        }
        self.reclaim();
        let picture = peek_info(access_unit)?;
        let (w, h, format) = (picture.width, picture.height, picture.format);
        let same = self
            .info
            .as_ref()
            .is_some_and(|i| (i.width, i.height, i.format) == (w, h, format));
        // A picture that then fails to decode leaves `info` as it was, so
        // the next picture of this size reports the new pool.
        let reconfigured = if same {
            None
        } else {
            Some(self.reconfigure(w, h, format)?)
        };
        let Some(slot) = self.free.pop() else {
            return Err(CodecError::PoolExhausted {
                held: self.pool.len() - self.free.len(),
                pool: self.pool.len(),
            });
        };
        let tensor = Arc::get_mut(&mut self.pool[slot as usize])
            .expect("a free slot's tensor has no other reference");
        if let Err(e) = self.decoder.decode_into_dyn(access_unit, tensor) {
            self.free.push(slot);
            return Err(e);
        }
        // The hardware writes straight into a tensor it can take as its
        // capture buffer; otherwise the picture is copied out of its own.
        let transfer = if self.decoder.v4l2_wrote_in_place() {
            Transfer::ZeroCopy
        } else {
            Transfer::CpuCopy
        };
        let mut events = Vec::new();
        let changed = reconfigured.or_else(|| self.info.clone().filter(|i| i.transfer != transfer));
        if let Some(mut info) = changed {
            info.transfer = transfer;
            self.info = Some(info.clone());
            events.push(DecodeEvent::Reconfigured(info));
        }
        events.push(DecodeEvent::Frame(DecodedFrame::new(
            Arc::clone(&self.pool[slot as usize]),
            pts,
            false,
            (self.generation, slot),
            Arc::clone(&self.returns),
        )));
        Ok(events)
    }

    fn flush(&mut self) -> Result<Vec<DecodeEvent>> {
        Ok(Vec::new())
    }

    fn stream_info(&self) -> Option<&StreamInfo> {
        self.info.as_ref()
    }

    fn device(&self) -> String {
        self.device.clone()
    }

    fn engine(&self) -> Engine {
        Engine::V4l2
    }
}
