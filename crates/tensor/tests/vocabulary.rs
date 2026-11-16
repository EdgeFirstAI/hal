//! Exercises `ef_vocabulary!` through the `#[doc(hidden)]` demo vocabulary,
//! and pins the codes and wire strings of `DType`, `PixelFormat` and
//! `TensorMemory`.

use edgefirst_tensor::vocabulary_demo::{demo_code, Demo};
use edgefirst_tensor::{
    ColorEncoding, ColorRange, ColorSpace, ColorTransfer, CpuAccess, DType, PixelFormat, Tensor,
    TensorMemory,
};

#[test]
fn code_and_from_code_round_trip_every_variant() {
    for v in [Demo::Alpha, Demo::Beta] {
        assert_eq!(Demo::from_code(v.code()), Some(v));
    }
}

#[test]
fn discriminants_are_the_declared_values_not_declaration_order() {
    assert_eq!(Demo::Beta.code(), 7);
    assert_eq!(demo_code::BETA, 7);
}

#[test]
fn wire_module_constants_match_code_for_every_variant() {
    // The wire constant and `.code()` agree.
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
    // Pins every `DType` code.
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
    // Pins every `PixelFormat` code.
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
    // Pins every `PixelFormat` wire string.
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
    // Pins every `TensorMemory` code. No variant is `cfg`-gated.
    let expect = [
        (TensorMemory::Mem, 0),
        (TensorMemory::Shm, 1),
        (TensorMemory::DmaBuf, 2),
        (TensorMemory::IoSurface, 3),
        (TensorMemory::Pbo, 4),
        (TensorMemory::Cuda, 5),
        (TensorMemory::D3d11Texture, 6),
    ];
    for (v, c) in expect {
        assert_eq!(v.code(), c, "{v:?}");
    }
    assert_eq!(
        TensorMemory::all().len(),
        7,
        "no variant may be cfg-gated away, and none added without pinning it"
    );
}

#[test]
fn every_tensor_memory_code_parses_on_every_platform() {
    for c in 0..7u32 {
        assert!(
            TensorMemory::from_code(c).is_some(),
            "code {c} must parse everywhere, whether or not it can be allocated here"
        );
    }
    assert_eq!(TensorMemory::from_code(7), None, "7 is not assigned");
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
        (TensorMemory::D3d11Texture, "d3d11"),
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
    // Asserts only that every variant can be queried, not a specific answer.
    for &v in TensorMemory::all() {
        let _ = v.is_available();
    }
}

#[test]
fn tensor_memory_wire_constants_match_code_for_every_variant() {
    // The const-only form and `.code()` agree.
    use edgefirst_tensor::tensor_memory_wire as w;
    assert_eq!(TensorMemory::Mem.code(), w::MEM);
    assert_eq!(TensorMemory::Shm.code(), w::SHM);
    assert_eq!(TensorMemory::DmaBuf.code(), w::DMABUF);
    assert_eq!(TensorMemory::IoSurface.code(), w::IOSURFACE);
    assert_eq!(TensorMemory::Pbo.code(), w::PBO);
    assert_eq!(TensorMemory::Cuda.code(), w::CUDA);
    assert_eq!(TensorMemory::D3d11Texture.code(), w::D3D11_TEXTURE);
}

#[test]
fn every_vocabulary_variant_round_trips_through_its_wire_string() {
    // Every variant's code and wire string round-trip and are unique.
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
    check(
        CpuAccess::all(),
        CpuAccess::code,
        CpuAccess::from_code,
        CpuAccess::as_str,
        CpuAccess::from_str_code,
    );
    check(
        ColorSpace::all(),
        ColorSpace::code,
        ColorSpace::from_code,
        ColorSpace::as_str,
        ColorSpace::from_str_code,
    );
    check(
        ColorTransfer::all(),
        ColorTransfer::code,
        ColorTransfer::from_code,
        ColorTransfer::as_str,
        ColorTransfer::from_str_code,
    );
    check(
        ColorEncoding::all(),
        ColorEncoding::code,
        ColorEncoding::from_code,
        ColorEncoding::as_str,
        ColorEncoding::from_str_code,
    );
    check(
        ColorRange::all(),
        ColorRange::code,
        ColorRange::from_code,
        ColorRange::as_str,
        ColorRange::from_str_code,
    );
}

#[test]
fn cpu_access_codes_and_strings_are_pinned() {
    use edgefirst_tensor::cpu_access_wire as w;
    let expect = [
        (CpuAccess::None, 0, w::NONE, "none"),
        (CpuAccess::Read, 1, w::READ, "read"),
        (CpuAccess::Write, 2, w::WRITE, "write"),
        (CpuAccess::ReadWrite, 3, w::READ_WRITE, "readwrite"),
    ];
    for (v, c, wire, s) in expect {
        assert_eq!(v.code(), c, "{v:?}");
        assert_eq!(wire, c, "{v:?}");
        assert_eq!(v.as_str(), s, "{v:?}");
        assert_eq!(CpuAccess::from_code(c), Some(v));
        assert_eq!(CpuAccess::from_str_code(s), Some(v));
    }
    assert_eq!(CpuAccess::all().len(), expect.len());
    assert_eq!(CpuAccess::from_code(4), None);
    assert_eq!(CpuAccess::default(), CpuAccess::ReadWrite);
}

#[test]
fn protocol_kind_is_the_tensor_memory_numbering() {
    // A descriptor's `kind` and `TensorMemory::code()` are one numbering.
    use edgefirst_tensor::protocol::kind;
    let expect = [
        (TensorMemory::Mem, kind::MEM),
        (TensorMemory::Shm, kind::SHM),
        (TensorMemory::DmaBuf, kind::DMABUF),
        (TensorMemory::IoSurface, kind::IOSURFACE),
        (TensorMemory::Pbo, kind::PBO),
        (TensorMemory::Cuda, kind::CUDA),
        (TensorMemory::D3d11Texture, kind::D3D11_TEXTURE),
    ];
    for (v, k) in expect {
        assert_eq!(v.code(), k, "{v:?}");
    }
    assert_eq!(TensorMemory::all().len(), expect.len());
}

#[test]
fn a_defined_but_unbacked_code_errors_instead_of_panicking() {
    // `IoSurface`, `Cuda` and `D3d11Texture` are codes with no backend:
    // requesting one returns an error.
    for v in [
        TensorMemory::IoSurface,
        TensorMemory::Cuda,
        TensorMemory::D3d11Texture,
    ] {
        let err = Tensor::<u8>::new(&[64], Some(v), None)
            .expect_err("no backend produces this code yet, so allocation cannot succeed");
        let msg = err.to_string();
        assert!(
            msg.contains("not supported by this build"),
            "{v:?} should report an unsupported backing, got: {msg}"
        );
    }
}
