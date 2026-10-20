// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Frames from the hardware video decoder convert on the GPU without a
//! copy: the decoder cycles a fixed pool of DMA-BUF tensors, so after one
//! pass over the pool every frame is an EGLImage cache hit. A decoded
//! picture is an NV12 tensor over the device's padded buffer (coded height
//! 1088 for a 1080 stream), so this also pins the plane offsets the decoder
//! describes against what the GPU reads.
//!
//! Skips (prints `SKIPPED`) without a decoder or GL, and without a DMA-BUF
//! heap unless `HAL_TEST_REQUIRE_DMA=1`.

#![cfg(all(target_os = "linux", feature = "opengl", feature = "codec"))]

use std::collections::BTreeSet;

use edgefirst_codec::video::{
    access_units, DecodeEvent, DecodedFrame, DecoderConfig, VideoDecoder,
};
use edgefirst_codec::CodecError;
use edgefirst_image::{GLProcessorThreaded, ImageProcessorTrait};
use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorDyn, TensorMemory};

#[path = "../../tensor/tests/support/dma_require.rs"]
mod dma_require;

const TEST: &str = "decoded_frames_convert_on_the_gpu_without_new_imports";

fn skip(why: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "WARNING: SKIPPED: {TEST} - {why}");
}

fn testdata(name: &str) -> Vec<u8> {
    let root = std::env::var("EDGEFIRST_TESTDATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("testdata")
        });
    let path = root.join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// The visible picture of an NV12 frame in a new contiguous DMA-BUF image.
fn contiguous_copy(src: &TensorDyn) -> TensorDyn {
    let (w, h) = (src.width().unwrap(), src.height().unwrap());
    let mut dst = TensorDyn::image(
        w,
        h,
        PixelFormat::Nv12,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    dst.set_colorimetry(src.colorimetry());
    let ds = dst.effective_row_stride().unwrap();
    let mut out = dst.map_bytes(CpuAccess::Write).unwrap();
    let ss = src.effective_row_stride().unwrap();
    let luma = src.map_bytes(CpuAccess::Read).unwrap();
    for y in 0..h {
        out[y * ds..y * ds + w].copy_from_slice(&luma[y * ss..y * ss + w]);
    }
    let chroma = src.chroma_dyn().unwrap();
    let cs = chroma.effective_row_stride().unwrap();
    let cmap = chroma.map_bytes(CpuAccess::Read).unwrap();
    for y in 0..h / 2 {
        let o = ds * h + y * ds;
        out[o..o + w].copy_from_slice(&cmap[y * cs..y * cs + w]);
    }
    drop(out);
    dst
}

fn rgba(w: usize, h: usize) -> TensorDyn {
    TensorDyn::image(
        w,
        h,
        PixelFormat::Rgba,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .unwrap()
}

/// Converts each decoded frame and records what the import cache did.
struct Check {
    gl: GLProcessorThreaded,
    dst: TensorDyn,
    frames: usize,
    buffers: usize,
    /// The pool tensors seen so far, by address.
    seen: BTreeSet<usize>,
    /// Frames whose buffer had carried an earlier frame.
    repeats: usize,
    /// Source imports made for those frames; zero-copy reuse makes none.
    reimports: u64,
    first_gl: Option<Vec<u8>>,
    first_copy: Option<Vec<u8>>,
}

impl Check {
    fn events(&mut self, events: Vec<DecodeEvent>) {
        for e in events {
            match e {
                DecodeEvent::Frame(f) => self.frame(&f),
                DecodeEvent::Reconfigured(info) => self.buffers = info.buffers,
            }
        }
    }

    fn frame(&mut self, frame: &DecodedFrame) {
        let src = frame.tensor();
        assert_eq!(src.memory(), TensorMemory::DmaBuf);
        let before = self.gl.egl_cache_stats().unwrap().src.misses;
        self.gl
            .convert(src, &mut self.dst, None)
            .expect("GL converts a decoded frame");
        let imports = self.gl.egl_cache_stats().unwrap().src.misses - before;
        if !self.seen.insert(src as *const TensorDyn as usize) {
            self.repeats += 1;
            self.reimports += imports;
        }
        if self.frames == 0 {
            self.first_gl = Some(self.dst.map_bytes(CpuAccess::Read).unwrap().to_vec());
            // The same pixels in a plain contiguous NV12 image, through the
            // same GPU path: any difference is the decoder's layout.
            let copy = contiguous_copy(src);
            let mut copy_dst = rgba(src.width().unwrap(), src.height().unwrap());
            self.gl.convert(&copy, &mut copy_dst, None).unwrap();
            self.first_copy = Some(copy_dst.map_bytes(CpuAccess::Read).unwrap().to_vec());
        }
        self.frames += 1;
    }
}

#[test]
fn decoded_frames_convert_on_the_gpu_without_new_imports() {
    if !dma_require::available_or_skip(edgefirst_tensor::is_dma_available(), TEST) {
        return;
    }
    let mut decoder = match VideoDecoder::new(DecoderConfig::h264()) {
        Ok(d) => d,
        Err(CodecError::NoDevice(why)) => return skip(&why),
        Err(e) => panic!("decoder open failed: {e}"),
    };
    // Both sources on the external-sampler path: the shader path filters
    // chroma differently at sharp edges, which would hide nothing about
    // the layout but fail the comparison. The only test in this binary.
    std::env::set_var("EDGEFIRST_NV_CONVERT_PATH", "sampler");
    let gl = match GLProcessorThreaded::new(None) {
        Ok(gl) => gl,
        Err(e) => return skip(&format!("GL not available: {e}")),
    };
    let (w, h) = (1920, 1080);

    let stream = testdata("h264_1080p_main_bframes.h264");
    let mut check = Check {
        gl,
        dst: rgba(w, h),
        frames: 0,
        buffers: 0,
        seen: BTreeSet::new(),
        repeats: 0,
        reimports: 0,
        first_gl: None,
        first_copy: None,
    };
    for (i, au) in access_units(&stream).enumerate() {
        check.events(decoder.decode(au, i as u64).unwrap());
    }
    check.events(decoder.flush().unwrap());
    let buffers = check.buffers;
    assert_eq!(check.frames, 30);
    assert!(
        buffers > 0 && buffers < check.frames,
        "pool of {buffers} for {} frames",
        check.frames
    );

    // A frame in a buffer that already carried one is a cache hit: decoded
    // pictures are imported once per pool buffer, never once per frame.
    assert!(
        check.seen.len() <= buffers,
        "frames came from outside the pool"
    );
    assert!(
        check.repeats >= check.frames - buffers,
        "{} of {} frames reused a buffer",
        check.repeats,
        check.frames
    );
    assert_eq!(
        check.reimports,
        0,
        "frames in reused buffers were imported again: {:?}",
        check.gl.egl_cache_stats().unwrap()
    );

    // The GPU read the same picture from the decoded frame as from its
    // contiguous copy: plane offsets and strides are right.
    let (gl_px, copy_px) = (check.first_gl.unwrap(), check.first_copy.unwrap());
    let stride = gl_px.len() / h;
    let mut worst = 0u8;
    let mut rows = Vec::new();
    for y in 0..h {
        let row_worst = (0..w * 4)
            .filter(|x| x % 4 != 3)
            .map(|x| gl_px[y * stride + x].abs_diff(copy_px[y * stride + x]))
            .max()
            .unwrap_or(0);
        if row_worst > 1 {
            rows.push((y, row_worst));
        }
        worst = worst.max(row_worst);
    }
    assert!(
        worst <= 1,
        "the decoded frame and its contiguous copy convert differently in {} rows \
         (worst {worst}); first {:?}, last {:?}",
        rows.len(),
        rows.first(),
        rows.last()
    );
}
