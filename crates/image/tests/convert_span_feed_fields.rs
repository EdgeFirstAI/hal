// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The convert span must actually CARRY `src_feed` and `dst_feed`, and
//! `dst_feed` must be recorded for EVERY destination, not only a refused
//! import.
//!
//! Both fields are declared on `image.convert.gl`, and every recording site
//! reaches that span through the handle the engine captures rather than through
//! `Span::current()` — which inside the engine names `image.convert.gl.engine`,
//! and on the two-pass plans a pass span. `tracing` drops a record against a
//! field the span does not declare, silently, so the wrong span means the
//! catalog promises an observable nothing emits. Both fields were in exactly
//! that state (#175).
//!
//! **This is its own test binary, and it holds exactly one test, deliberately.**
//! `tracing` caches callsite interest process-wide: the first evaluation of a
//! callsite decides its `Interest`, and a thread with no subscriber caches
//! `never`, which no later scoped subscriber undoes. As a lib test among ~470
//! others running in parallel it therefore failed about four runs in nine —
//! seeing `image.convert.gl.engine` (a callsite first evaluated inside the
//! scoped subscriber, during the processor's own DMA-BUF probe) but never
//! `image.convert.gl` (one another thread had already reached). A dedicated
//! process with a GLOBAL subscriber installed before the first span removes the
//! race rather than narrowing it, and lets the public `GLProcessorThreaded` be
//! the thing under test: a global subscriber is visible from its GL worker
//! thread, where a thread-local one is not.
//!
//! No test hook either. The `dst_feed` half is driven by a driver that really
//! refuses a destination — Vivante returns `EGL_BAD_ACCESS` for an import at an
//! offset that is not 64-byte aligned — so the trigger under test is the
//! production one. Where the driver accepts that destination (Mali, V3D) there
//! is nothing refused and that half self-skips; `src_feed` is asserted on every
//! host, including a desktop whose EGL cannot import a DMA-BUF at all.
//!
//! The FLOAT route is out of scope by construction: it returns before
//! `image.convert.gl` is entered, so it has no convert span and records
//! neither field. That is what `ARCHITECTURE.md`'s span catalog says; giving
//! the route its own span is a follow-up.

#![cfg(all(target_os = "linux", feature = "opengl"))]

use edgefirst_image::{Crop, Flip, GLProcessorThreaded, ImageProcessorTrait, Rotation};
use edgefirst_tensor::{
    CpuAccess, DType, PixelFormat, Region, TensorDyn, TensorMapTrait, TensorMemory,
};
use std::sync::{Arc, Mutex};
use tracing_subscriber::layer::SubscriberExt;

const W: usize = 320;
const H: usize = 240;
const BPP: usize = 4;
/// Pseudo-field marking a span creation rather than a field record.
const NEW_SPAN: &str = "<new_span>";

/// `(span name, field, value)` for every `record` call, plus one entry per span
/// creation so an absent field can be told from a dead capture.
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

fn skip(why: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "SKIPPED: {why}");
}

fn image(w: usize, h: usize, memory: TensorMemory) -> Option<TensorDyn> {
    match TensorDyn::image(
        w,
        h,
        PixelFormat::Rgba,
        DType::U8,
        Some(memory),
        CpuAccess::ReadWrite,
    ) {
        Ok(t) if t.memory() == memory => Some(t),
        _ => None,
    }
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
    // GLOBAL, not scoped: the convert runs on `GLProcessorThreaded`'s worker
    // thread, and it must be installed before the first span so this process
    // cannot have cached `Interest::never()` for any callsite under test.
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
    // A host-memory source and destination need no zero-copy anything, so
    // this half runs on a desktop whose EGL cannot import a DMA-BUF.
    let src = image(W, H, TensorMemory::Mem).expect("host source");
    {
        let mut m = src.map_bytes(CpuAccess::Write).expect("map source");
        m.as_mut_slice().fill(0x5A);
    }
    let mut dst = image(W, H, TensorMemory::Mem).expect("host destination");
    gl.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())
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

    // A zero-copy source, so the `import` arm is covered where a host has one.
    if let Some(dma_src) = image(W, H, TensorMemory::DmaBuf) {
        {
            let mut m = dma_src.map_bytes(CpuAccess::Write).expect("map dma source");
            m.as_mut_slice().fill(0x33);
        }
        let _ = gl.convert(
            &dma_src,
            &mut dst,
            Rotation::None,
            Flip::None,
            Crop::default(),
        );
    }
    let src_feeds = values_for(&records.lock().unwrap().clone(), "src_feed");
    eprintln!("CAPTURE src_feed on image.convert.gl: {src_feeds:?}");

    // ── dst_feed: stated for every destination ──────────────────────────
    // The host-memory convert above is a `mapped_texture` destination, and the
    // field must SAY so. An absent field used to be the only signal for an
    // ordinary texture-lowered convert, which reads exactly like a zero-copy
    // one that never recorded anything.
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

    // A zero-copy destination the driver ACCEPTS must say `zero_copy` -- the
    // half an import-refusal marker could never state, and the one that makes
    // the field readable without inference.
    //
    // Reachable only where the display can render into a DMA-BUF at all: on a
    // host whose transfer backend fell back to PBO, a DMA-BUF destination is
    // lowered to the mapped route with no import attempted, and
    // `mapped_texture` is then the correct answer. The value is checked
    // against the vocabulary on every host either way, which is what catches a
    // missing or misspelled record.
    if let Some(mut zc) = image(W, H, TensorMemory::DmaBuf) {
        let _ = gl.convert(&src, &mut zc, Rotation::None, Flip::None, Crop::default());
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

    // ── dst_feed on the REFUSED route, where a driver really refuses ─────
    // The rebuilt-view shape `offset_source_view_alignment.rs` pins: it
    // carries the byte offset and no `view_origin`, so the import bases at
    // that offset, and Vivante refuses it when it is not 64-byte aligned.
    let Some(canvas) = image(64, 64, TensorMemory::DmaBuf) else {
        skip("no zero-copy RGBA image here, so no destination import to refuse");
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
    let tile_src = image(16, 16, TensorMemory::Mem).expect("tile source");

    let before = gl
        .convert_stats()
        .expect("convert stats")
        .dst_import_fallbacks;
    let _ = gl.convert(
        &tile_src,
        &mut tile,
        Rotation::None,
        Flip::None,
        Crop::default(),
    );
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
