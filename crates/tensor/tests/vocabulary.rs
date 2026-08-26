//! Exercises `ef_vocabulary!` through the `#[doc(hidden)]` demo vocabulary
//! rather than a real one -- the first few tests only prove the mechanism:
//! explicit, non-contiguous discriminants round-trip through the code, the
//! string form, and the generated wire module. `DType`, `PixelFormat` and
//! `TensorMemory` (below) are converted onto the macro directly. `DType`
//! and `PixelFormat` had values that already shipped, so their tests pin
//! the macro's output against those; `TensorMemory` had three disagreeing
//! numberings and no shipped Rust one, so its test pins the canonical
//! assignment this vocabulary establishes.

use edgefirst_tensor::vocabulary_demo::{demo_code, Demo};
use edgefirst_tensor::{DType, PixelFormat, Tensor, TensorMemory};

#[test]
fn code_and_from_code_round_trip_every_variant() {
    for v in [Demo::Alpha, Demo::Beta] {
        assert_eq!(Demo::from_code(v.code()), Some(v));
    }
}

#[test]
fn discriminants_are_the_declared_values_not_declaration_order() {
    // The bug this macro exists to prevent: implicit ordering means a
    // cfg-gated variant in the middle shifts everything after it.
    assert_eq!(Demo::Beta.code(), 7);
    assert_eq!(demo_code::BETA, 7);
}

#[test]
fn wire_module_constants_match_code_for_every_variant() {
    // The property that actually matters: the free-standing wire constant
    // and `.code()` are generated from the same declared literal, so a
    // const-only consumer (cbindgen, FFI) and the Rust API cannot silently
    // disagree about a variant's number.
    assert_eq!(Demo::Alpha.code(), demo_code::ALPHA);
    assert_eq!(Demo::Beta.code(), demo_code::BETA);
}

#[test]
fn an_unassigned_code_is_rejected_not_guessed() {
    assert_eq!(Demo::from_code(999), None);
}

#[test]
fn str_code_round_trips_and_rejects_unknown_strings() {
    for v in [Demo::Alpha, Demo::Beta] {
        assert_eq!(Demo::from_str_code(v.as_str()), Some(v));
    }
    assert_eq!(Demo::from_str_code("gamma"), None);
}

#[test]
fn all_lists_every_variant_exactly_once() {
    assert_eq!(Demo::all(), &[Demo::Alpha, Demo::Beta]);
}

#[test]
fn dtype_codes_are_unchanged_by_the_macro() {
    // Pinned against the values that ship today (protocol.rs `mod dtype`,
    // hal.h `hal_dtype`). If the macro changes any of these, it has silently
    // broken a wire format and a C ABI at once.
    let expect = [
        (DType::U8, 0),
        (DType::I8, 1),
        (DType::U16, 2),
        (DType::I16, 3),
        (DType::U32, 4),
        (DType::I32, 5),
        (DType::U64, 6),
        (DType::I64, 7),
        (DType::F16, 8),
        (DType::F32, 9),
        (DType::F64, 10),
    ];
    for (v, c) in expect {
        assert_eq!(v.code(), c, "{v:?}");
    }
    assert_eq!(
        DType::all().len(),
        expect.len(),
        "a variant was added without pinning it"
    );
}

#[test]
fn pixel_format_codes_are_unchanged_by_the_macro() {
    // Pinned against the values that ship today (protocol.rs `mod format`,
    // the Rust `#[repr(u8)]` discriminants that predated the macro, and
    // Python). If the macro changes any of these, it has silently broken a
    // wire format and cross-package value equality at once. `hal_pixel_format`
    // is deliberately NOT pinned here -- it is the one surface that keeps its
    // own, differently-numbered values; see `to_hal_pixel_format` in the capi
    // crate.
    let expect = [
        (PixelFormat::Rgb, 1),
        (PixelFormat::Rgba, 2),
        (PixelFormat::Bgra, 3),
        (PixelFormat::Grey, 4),
        (PixelFormat::Yuyv, 5),
        (PixelFormat::Vyuy, 6),
        (PixelFormat::Nv12, 7),
        (PixelFormat::Nv16, 8),
        (PixelFormat::PlanarRgb, 9),
        (PixelFormat::PlanarRgba, 10),
        (PixelFormat::Nv24, 11),
    ];
    for (v, c) in expect {
        assert_eq!(v.code(), c, "{v:?}");
    }
    assert_eq!(
        PixelFormat::all().len(),
        expect.len(),
        "a variant was added without pinning it"
    );
}

#[test]
fn pixel_format_wire_strings_match_the_hal_format_table() {
    // Pinned against `format.rs`'s doc comment, which is the format table
    // `Tensor.msg` in the schemas repo cites as canonical. These strings are
    // deliberately NOT `Display` (FourCC-derived) or the `serde` output
    // (PascalCase variant names) -- see that doc comment for why unifying
    // them would be a breaking change to two formats that are not this one.
    let expect = [
        (PixelFormat::Rgb, "rgb8"),
        (PixelFormat::Rgba, "rgba8"),
        (PixelFormat::Bgra, "bgra8"),
        (PixelFormat::Grey, "mono8"),
        (PixelFormat::Yuyv, "YUYV"),
        (PixelFormat::Vyuy, "VYUY"),
        (PixelFormat::Nv12, "NV12"),
        (PixelFormat::Nv16, "NV16"),
        (PixelFormat::PlanarRgb, "rgb8_planar"),
        (PixelFormat::PlanarRgba, "rgba8_planar"),
        (PixelFormat::Nv24, "NV24"),
    ];
    for (v, s) in expect {
        assert_eq!(v.as_str(), s, "{v:?}");
    }
    assert_eq!(
        PixelFormat::all().len(),
        expect.len(),
        "a variant was added without pinning its wire string"
    );
}

#[test]
fn every_pixel_format_round_trips_through_its_string() {
    for &v in PixelFormat::all() {
        assert_eq!(PixelFormat::from_str_code(v.as_str()), Some(v), "{v:?}");
    }
}

#[test]
fn tensor_memory_defines_every_variant_on_every_platform() {
    // A vocabulary is a namespace of codes, not a capability list. A blob
    // recorded on Linux and replayed on macOS carries storage kind
    // "dmabuf"; the consumer must be able to NAME it to report "cannot
    // materialise here" instead of failing with an unknown code. That is
    // why no variant here is `cfg`-gated -- `Shm` used to be, which made
    // every variant after it shift by one on a non-unix build.
    let expect = [
        (TensorMemory::Mem, 0),
        (TensorMemory::Shm, 1),
        (TensorMemory::DmaBuf, 2),
        (TensorMemory::IoSurface, 3),
        (TensorMemory::Pbo, 4),
        (TensorMemory::Cuda, 5),
    ];
    for (v, c) in expect {
        assert_eq!(v.code(), c, "{v:?}");
    }
    assert_eq!(
        TensorMemory::all().len(),
        6,
        "no variant may be cfg-gated away, and none added without pinning it"
    );
}

#[test]
fn every_tensor_memory_code_parses_on_every_platform() {
    for c in 0..6u32 {
        assert!(
            TensorMemory::from_code(c).is_some(),
            "code {c} must parse everywhere, whether or not it can be allocated here"
        );
    }
    assert_eq!(TensorMemory::from_code(6), None, "6 is not assigned");
}

#[test]
fn tensor_memory_wire_strings_round_trip() {
    let expect = [
        (TensorMemory::Mem, "mem"),
        (TensorMemory::Shm, "shm"),
        (TensorMemory::DmaBuf, "dmabuf"),
        (TensorMemory::IoSurface, "iosurface"),
        (TensorMemory::Pbo, "pbo"),
        (TensorMemory::Cuda, "cuda"),
    ];
    for (v, s) in expect {
        assert_eq!(v.as_str(), s, "{v:?}");
        assert_eq!(TensorMemory::from_str_code(s), Some(v), "{s}");
    }
    assert_eq!(
        TensorMemory::all().len(),
        expect.len(),
        "a variant was added without pinning its wire string"
    );
}

#[test]
fn tensor_memory_availability_is_a_runtime_question_not_a_compile_time_one() {
    // Never asserts a specific answer -- it depends on the host's dma_heap
    // permissions, whether a GL context is current, and whether libcuda is
    // installed. It asserts only that ASKING is possible for every variant,
    // including ones this platform cannot allocate: a variant that is
    // merely unavailable must still be nameable and queryable, never a
    // compile error or a panic.
    for &v in TensorMemory::all() {
        let _ = v.is_available();
    }
}

#[test]
fn tensor_memory_wire_constants_match_code_for_every_variant() {
    // The const-only form (for cbindgen and match-free FFI) and `.code()`
    // are generated from the same declared literal, so they cannot drift.
    use edgefirst_tensor::tensor_memory_wire as w;
    assert_eq!(TensorMemory::Mem.code(), w::MEM);
    assert_eq!(TensorMemory::Shm.code(), w::SHM);
    assert_eq!(TensorMemory::DmaBuf.code(), w::DMABUF);
    assert_eq!(TensorMemory::IoSurface.code(), w::IOSURFACE);
    assert_eq!(TensorMemory::Pbo.code(), w::PBO);
    assert_eq!(TensorMemory::Cuda.code(), w::CUDA);
}

#[test]
fn every_vocabulary_variant_round_trips_through_its_wire_string() {
    // The wire strings are a public contract -- `format` is a string on the
    // wire -- and unlike the numbers they have no compile-time link to any
    // other surface at all. The pinning tests above name the exact spelling
    // of each; this one is driven by `all()`, so a variant added later is
    // covered without anyone remembering to extend a list.
    //
    // Uniqueness is the property being checked, not just round-tripping:
    // two variants sharing a string would make `from_str_code` pick one and
    // silently mistranslate the other, and the same holds for the codes.
    fn check<V: Copy + std::fmt::Debug + PartialEq>(
        all: &[V],
        code: impl Fn(V) -> u32,
        from_code: impl Fn(u32) -> Option<V>,
        as_str: impl Fn(V) -> &'static str,
        from_str: impl Fn(&str) -> Option<V>,
    ) {
        let mut seen_codes = std::collections::BTreeSet::new();
        let mut seen_strs = std::collections::BTreeSet::new();
        for &v in all {
            assert_eq!(from_str(as_str(v)), Some(v), "{v:?} does not round-trip");
            assert_eq!(from_code(code(v)), Some(v), "{v:?} does not round-trip");
            assert!(seen_codes.insert(code(v)), "{v:?} reuses code {}", code(v));
            assert!(
                seen_strs.insert(as_str(v)),
                "{v:?} reuses wire string {:?}",
                as_str(v)
            );
        }
        assert!(!all.is_empty(), "an empty vocabulary would pass vacuously");
    }

    check(
        DType::all(),
        DType::code,
        DType::from_code,
        DType::as_str,
        DType::from_str_code,
    );
    check(
        PixelFormat::all(),
        PixelFormat::code,
        PixelFormat::from_code,
        PixelFormat::as_str,
        PixelFormat::from_str_code,
    );
    check(
        TensorMemory::all(),
        TensorMemory::code,
        TensorMemory::from_code,
        TensorMemory::as_str,
        TensorMemory::from_str_code,
    );
}

#[test]
fn tensor_memory_is_not_protocol_kind() {
    // Two vocabularies that describe overlapping things and must not be
    // bridged by integer. `kind` records what a `TensorDesc`'s `ptr` and
    // `handle` MEAN, so it collapses Mem and Shm into one code; passing a
    // `TensorMemory::code()` where a `kind` is expected would read Shm(1)
    // as DMABUF(1). `protocol::kind_of` is the only legitimate bridge.
    use edgefirst_tensor::protocol::kind;
    assert_ne!(TensorMemory::DmaBuf.code(), kind::DMABUF);
    assert_ne!(TensorMemory::Pbo.code(), kind::PBO);
    assert_eq!(
        TensorMemory::Shm.code(),
        kind::DMABUF,
        "the collision this test exists to document: Shm's code equals the DMABUF kind"
    );
}

#[test]
fn a_defined_but_unbacked_code_errors_instead_of_panicking() {
    // `IoSurface` and `Cuda` are codes with no backend behind them yet, so
    // every match arm handling them is unreachable *today* -- which is
    // exactly when `unreachable!()` reads as correct and becomes a panic in
    // library code the day Plan 3 makes a backend report one. Pinning a
    // request to them must come back as an error a caller can handle.
    for v in [TensorMemory::IoSurface, TensorMemory::Cuda] {
        let err = Tensor::<u8>::new(&[64], Some(v), None)
            .expect_err("no backend produces this code yet, so allocation cannot succeed");
        let msg = err.to_string();
        assert!(
            msg.contains("not supported by this build"),
            "{v:?} should report an unsupported backing, got: {msg}"
        );
    }
}
