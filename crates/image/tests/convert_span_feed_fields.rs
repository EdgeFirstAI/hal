// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The `image.convert.gl` span carries `src_feed` and `dst_feed`, and
//! `dst_feed` is recorded for every destination.
//!
//! This test binary holds one test and installs a global `tracing` subscriber
//! before the first span. The refused-destination and float halves skip on a
//! host that cannot reach them.

#![cfg(all(feature = "opengl", any(target_os = "linux", target_os = "windows")))]

use edgefirst_image::{GLProcessorThreaded, ImageProcessorTrait};
use edgefirst_tensor::{
    CpuAccess, DType, PixelFormat, Region, TensorDyn, TensorMapTrait, TensorMemory,
};
use std::sync::{Arc, Mutex};
use tracing_subscriber::layer::SubscriberExt;

const W: usize = 320;
const H: usize = 240;
const BPP: usize = 4;
/// Pseudo-field marking a span creation.
const NEW_SPAN: &str = "<new_span>";

/// `(span name, field, value)` for every `record` call, plus one entry per span creation.
type Records = Arc<Mutex<Vec<(String, String, String)>>>;

struct Capture(Records);
struct Visit<'a>(&'a str, &'a Records);

impl tracing::field::Visit for Visit<'_> {
    fn record_debug(&mut self, f: &tracing::field::Field, v: &dyn std::fmt::Debug) {
        self.1
            .lock()
            .unwrap()
            .push((self.0.to_string(), f.name().to_string(), format!("{v:?}")));
    }
    fn record_str(&mut self, f: &tracing::field::Field, v: &str) {
        self.1
            .lock()
            .unwrap()
            .push((self.0.to_string(), f.name().to_string(), v.to_string()));
    }
}

impl<S> tracing_subscriber::Layer<S> for Capture
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        _id: &tracing::span::Id,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        self.0.lock().unwrap().push((
            attrs.metadata().name().to_string(),
            NEW_SPAN.to_string(),
            String::new(),
        ));
    }

    fn on_record(
        &self,
        id: &tracing::span::Id,
        values: &tracing::span::Record<'_>,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let name = ctx
            .span(id)
            .map(|s| s.name().to_string())
            .unwrap_or_default();
        values.record(&mut Visit(&name, &self.0));
    }
}

#[path = "../../tensor/tests/support/dma_require.rs"]
#[allow(dead_code)]
mod dma_require;
#[path = "../../tensor/tests/support/outcome.rs"]
#[allow(dead_code)]
mod outcome;
#[path = "support/zero_copy.rs"]
#[allow(dead_code)]
mod zero_copy;
use outcome::report_skip as skip;

fn host_image(w: usize, h: usize) -> TensorDyn {
    TensorDyn::image(
        w,
        h,
        PixelFormat::Rgba,
        DType::U8,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("host-memory RGBA image")
}

fn zero_copy_rgba(w: usize, h: usize) -> Option<TensorDyn> {
    zero_copy::image(w, h, PixelFormat::Rgba, DType::U8)
}

/// Values recorded for `field` on the `image.convert.gl` span.
fn values_for(seen: &[(String, String, String)], field: &str) -> Vec<String> {
    seen.iter()
        .filter(|(span, f, _)| span == "image.convert.gl" && f == field)
        .map(|(_, _, v)| v.clone())
        .collect()
}

#[test]
fn the_convert_span_records_the_feed_fields() {
    let records: Records = Arc::new(Mutex::new(Vec::new()));
    // A global subscriber, installed before the first span.
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(Capture(records.clone())),
    )
    .expect("install the capturing subscriber");

    let mut gl = match GLProcessorThreaded::new(None) {
        Ok(gl) => gl,
        Err(e) => {
            assert!(
                !std::env::var("HAL_TEST_REQUIRE_GL").is_ok_and(|v| v == "1"),
                "HAL_TEST_REQUIRE_GL=1 but the GL backend failed to come up: {e}"
            );
            skip(&format!("no GL backend: {e}"));
            return;
        }
    };

    // ── src_feed: asserted on every host ────────────────────────────────
    let src = host_image(W, H);
    {
        let mut m = src.map_bytes(CpuAccess::Write).expect("map source");
        m.as_mut_slice().fill(0x5A);
    }
    let mut dst = host_image(W, H);
    gl.convert(&src, &mut dst, None)
        .expect("host-memory convert");

    let seen = records.lock().unwrap().clone();
    assert!(
        seen.iter()
            .any(|(span, field, _)| span == "image.convert.gl" && field == NEW_SPAN),
        "the capture never saw an image.convert.gl span, so it cannot testify about \
         the feed fields; records seen: {seen:?}"
    );
    let feeds = values_for(&seen, "src_feed");
    assert!(
        !feeds.is_empty(),
        "a u8 convert recorded no src_feed on image.convert.gl; records seen: {seen:?}"
    );
    assert!(
        feeds
            .iter()
            .all(|v| matches!(v.as_str(), "import" | "pbo" | "upload")),
        "src_feed must be one of import|pbo|upload, saw {feeds:?}"
    );

    // A zero-copy source, where the host has one.
    if let Some(dma_src) = zero_copy_rgba(W, H) {
        {
            let mut m = dma_src.map_bytes(CpuAccess::Write).expect("map dma source");
            m.as_mut_slice().fill(0x33);
        }
        let _ = gl.convert(&dma_src, &mut dst, None);
    }
    let src_feeds = values_for(&records.lock().unwrap().clone(), "src_feed");
    eprintln!("CAPTURE src_feed on image.convert.gl: {src_feeds:?}");

    // ── dst_feed: stated for every destination ──────────────────────────
    let dst_feeds = values_for(&records.lock().unwrap().clone(), "dst_feed");
    assert!(
        dst_feeds.iter().any(|v| v == "mapped_texture"),
        "a host-memory destination recorded no dst_feed; values seen: {dst_feeds:?}"
    );
    assert!(
        dst_feeds
            .iter()
            .all(|v| matches!(v.as_str(), "zero_copy" | "mapped_texture" | "pbo")),
        "dst_feed must be one of zero_copy|mapped_texture|pbo, saw {dst_feeds:?}"
    );

    // A zero-copy destination, where the host has one.
    if let Some(mut zc) = zero_copy_rgba(W, H) {
        let _ = gl.convert(&src, &mut zc, None);
    }
    let feeds = values_for(&records.lock().unwrap().clone(), "dst_feed");
    assert!(
        feeds
            .iter()
            .all(|v| matches!(v.as_str(), "zero_copy" | "mapped_texture" | "pbo")),
        "dst_feed must be one of zero_copy|mapped_texture|pbo, saw {feeds:?}"
    );
    if feeds.iter().any(|v| v == "zero_copy") {
        eprintln!("CAPTURE dst_feed zero_copy: present");
    } else {
        skip("no zero-copy destination render on this host, so dst_feed=zero_copy is unreachable");
    }
    eprintln!("CAPTURE dst_feed values: {feeds:?}");

    // ── Float route: same span, both feed fields ────────────────────────
    // The float destination is the platform's zero-copy GPU buffer.
    #[cfg(target_os = "linux")]
    let (dst_fmt, dst_dtype) = (PixelFormat::PlanarRgb, DType::F16);
    #[cfg(target_os = "windows")]
    let (dst_fmt, dst_dtype) = (PixelFormat::Rgba, DType::F32);
    if let Some(mut float_dst) = zero_copy::image(W, H, dst_fmt, dst_dtype) {
        let float_src = host_image(W, H);
        {
            let mut m = float_src
                .map_bytes(CpuAccess::Write)
                .expect("map float src");
            m.as_mut_slice().fill(0x40);
        }
        records.lock().unwrap().clear();
        match gl.convert(&float_src, &mut float_dst, None) {
            Ok(()) => {
                let seen = records.lock().unwrap().clone();
                let src_feeds = values_for(&seen, "src_feed");
                let dst_feeds = values_for(&seen, "dst_feed");
                assert!(
                    src_feeds
                        .iter()
                        .any(|v| matches!(v.as_str(), "import" | "pbo" | "upload")),
                    "a float convert recorded no src_feed on image.convert.gl; records: {seen:?}"
                );
                assert!(
                    dst_feeds
                        .iter()
                        .any(|v| matches!(v.as_str(), "zero_copy" | "pbo")),
                    "a float convert recorded no dst_feed on image.convert.gl; records: {seen:?}"
                );
                eprintln!(
                        "CAPTURE float src_feed={src_feeds:?} dst_feed={dst_feeds:?} for {dst_fmt:?}/{dst_dtype:?}"
                    );
            }
            Err(e) => {
                skip(&format!(
                    "GL float convert not routable on this host ({dst_fmt:?}/{dst_dtype:?}): {e}"
                ));
            }
        }
    }

    // ── dst_feed on the refused route ───────────────────────────────────
    // A destination rebuilt from a view's descriptor at a byte offset that is
    // not 64-byte aligned.
    let Some(canvas) = zero_copy_rgba(64, 64) else {
        return;
    };
    let pitch = canvas.effective_row_stride().unwrap_or(64 * BPP);
    let offset = 8 * pitch + 8 * BPP;
    assert!(
        !offset.is_multiple_of(64),
        "precondition: (8,8) at pitch {pitch} is byte offset {offset}, which must be \
         unaligned for a driver to have anything to refuse"
    );
    let fresh = canvas
        .view(Region::new(8, 8, 16, 16))
        .expect("destination view");
    let mut tile = TensorDyn::import_descriptor(&fresh.descriptor_pinned(None))
        .expect("rebuild the destination view from its descriptor");
    tile.set_plane_offset(offset);
    let tile_src = host_image(16, 16);

    let before = gl
        .convert_stats()
        .expect("convert stats")
        .dst_import_fallbacks;
    let _ = gl.convert(&tile_src, &mut tile, None);
    let refused = gl
        .convert_stats()
        .expect("convert stats")
        .dst_import_fallbacks
        - before;
    if refused == 0 {
        skip(&format!(
            "no destination import was refused at offset {offset} — either this \
             driver accepts it (Mali, V3D) or this host attempts no zero-copy \
             destination import at all — so there is no dst_feed to observe"
        ));
        return;
    }
    let seen = records.lock().unwrap().clone();
    assert!(
        values_for(&seen, "dst_feed")
            .iter()
            .any(|v| v == "mapped_texture"),
        "the refused destination import recorded no dst_feed on image.convert.gl; \
         records seen: {seen:?}"
    );
    eprintln!("CAPTURE dst_feed on the refused route: mapped_texture");
}
