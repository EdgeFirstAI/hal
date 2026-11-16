//! PNG feature gating.
//!
//! Format detection stays compiled in when the decoder is gated out: a valid
//! PNG reports `PngNotEnabled`, through both `peek_info` and
//! `decode_from_reader`.

use edgefirst_codec::peek_info;

fn png_bytes() -> Vec<u8> {
    // `EDGEFIRST_TESTDATA_DIR` when set, else a CWD-relative path.
    let root = std::env::var("EDGEFIRST_TESTDATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
                .join("testdata")
        });
    std::fs::read(root.join("zidane_exif_1.png")).expect("fixture")
}

#[cfg(feature = "png")]
#[test]
fn png_with_feature_is_recognised() {
    let info = peek_info(&png_bytes()).expect("valid PNG header");
    assert!(info.width > 0 && info.height > 0);
}

#[cfg(not(feature = "png"))]
#[test]
fn peek_reports_disabled_not_corrupt() {
    use edgefirst_codec::{CodecError, UnsupportedFeature};
    match peek_info(&png_bytes()) {
        Err(CodecError::Unsupported(UnsupportedFeature::PngNotEnabled)) => {}
        other => panic!("expected PngNotEnabled, got {other:?}"),
    }
}

#[cfg(not(feature = "png"))]
#[test]
fn reader_path_also_reports_disabled() {
    use edgefirst_codec::{CodecError, ImageDecoder, ImageLoad, UnsupportedFeature};
    use edgefirst_tensor::{CpuAccess, PixelFormat, Tensor, TensorMemory};

    let mut t = Tensor::<u8>::image(
        64,
        64,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::Write,
    )
    .expect("alloc");
    let mut d = ImageDecoder::new();
    let data = png_bytes();
    match t.load_image_read(&mut d, data.as_slice()) {
        Err(CodecError::Unsupported(UnsupportedFeature::PngNotEnabled)) => {}
        other => panic!("reader path must report PngNotEnabled, got {other:?}"),
    }
}
