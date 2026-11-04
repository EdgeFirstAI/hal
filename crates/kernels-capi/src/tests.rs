// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::ffi::{c_void, CStr};
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

use edgefirst_kernels::Pool;

use crate::*;
use edgefirst_kernels as k;

const I8: u32 = EfKernelDtype::I8 as u32;
const U8: u32 = EfKernelDtype::U8 as u32;
const I16: u32 = EfKernelDtype::I16 as u32;
const F32: u32 = EfKernelDtype::F32 as u32;

fn header_text() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/include/edgefirst/kernels.h"
    ))
    .expect("cbindgen wrote include/edgefirst/kernels.h")
}

/// Syntax-checks `src` against the header with `-Werror`, so a warning fails too.
fn cc_syntax_check(src: &str) -> Option<std::process::Output> {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    match std::process::Command::new(&cc)
        .args([
            "-fsyntax-only",
            "-Werror",
            "-Wall",
            "-I",
            concat!(env!("CARGO_MANIFEST_DIR"), "/include"),
            src,
        ])
        .output()
    {
        Ok(o) => Some(o),
        Err(e) => {
            use std::io::Write;
            let _ = writeln!(
                std::io::stderr(),
                "WARNING: SKIPPED: no C compiler ({cc}: {e})"
            );
            None
        }
    }
}

#[test]
fn the_generated_header_exists_and_declares_the_abi_probe() {
    assert!(header_text().contains("ef_kernels_abi_version"));
}

#[test]
fn the_header_carries_none_of_the_monoliths_symbols() {
    assert!(
        !header_text().contains("hal_"),
        "kernels.h must not carry the monolith's hal_* symbols"
    );
}

#[test]
fn no_rust_type_name_leaks_into_the_c_api() {
    let h = header_text();
    for rust_name in [
        "EfSoftmaxI8Plan",
        "EfSoftmaxI16Plan",
        "EfSigmoidPlan",
        "EfDflPlan",
        "EfDist2bboxPlan",
        "EfDflBoxPlan",
        "EfLutI8Plan",
        "EfBinaryI8Plan",
        "EfMatmulI8Plan",
        "EfSoftmaxDotI8Plan",
        "EfAttentionI8Plan",
        "EfAttentionScratch",
        "EfAttentionQMul",
        "EfTfliteKernels",
        "EfBinaryOp",
        "EfKernel",
        "BinaryI8Plan",
        "MatMulI8Plan",
        "AttentionScratch",
        "SoftmaxI8Plan",
        "DflBoxPlan",
    ] {
        assert!(
            !h.contains(rust_name),
            "the Rust type name `{rust_name}` leaked into kernels.h; add it to \
             cbindgen.toml's [export.rename]"
        );
    }
}

#[test]
fn the_header_survives_being_included_twice() {
    let src = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/c/test_double_include.c");
    let Some(out) = cc_syntax_check(src) else {
        return;
    };
    assert!(
        out.status.success(),
        "kernels.h failed to compile when included twice:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn the_shipped_artifact_is_libedgefirst_kernels_not_capi() {
    let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    dir.push("target");
    dir.push("debug");
    let found: Vec<String> = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|n| n.starts_with("libedgefirst_kernels"))
                .collect()
        })
        .unwrap_or_default();
    if found.is_empty() {
        use std::io::Write;
        let _ = writeln!(
            std::io::stderr(),
            "WARNING: SKIPPED: nothing built; name not checked"
        );
        return;
    }
    assert!(
        !found.iter().any(|n| n.contains("_capi")),
        "must ship as libedgefirst_kernels.*, never ..._capi; found {found:?}. \
         If `[lib] name` is right these are STALE artifacts from a previous \
         name — cargo does not remove them on rename."
    );
}

fn q(scale: f32, zero_point: i32) -> EfKernelQuant {
    EfKernelQuant { scale, zero_point }
}

fn out_q() -> EfKernelQuant {
    q(1.0 / 256.0, -128)
}

fn view<T>(data: &mut [T], dtype: u32, shape: &[usize]) -> EfKernelView {
    EfKernelView {
        data: data.as_mut_ptr().cast(),
        dtype,
        ndim: shape.len(),
        shape: shape.as_ptr(),
        strides: ptr::null(),
    }
}

fn errno() -> i32 {
    errno::errno().0
}

fn last_error() -> Option<String> {
    let p = ef_kernels_last_error();
    // SAFETY: a non-null result is a NUL-terminated string valid until the next call.
    (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
}

fn softmax_i8_params() -> EfSoftmaxI8Params {
    EfSoftmaxI8Params {
        input: q(0.1, 0),
        output: out_q(),
        beta: 1.0,
    }
}

#[test]
fn abi_version_is_one() {
    assert_eq!(ef_kernels_abi_version(), 1);
}

#[test]
fn last_error_is_null_on_a_fresh_thread() {
    std::thread::spawn(|| assert!(ef_kernels_last_error().is_null()))
        .join()
        .unwrap();
}

#[test]
fn null_args_return_einval() {
    let params = softmax_i8_params();
    let mut plan: *mut EfSoftmaxI8Plan = ptr::null_mut();
    // SAFETY: null arguments are the point of the test.
    unsafe {
        assert_eq!(ef_softmax_i8_plan_new(ptr::null(), &mut plan), -1);
        assert_eq!(errno(), libc::EINVAL);
        assert!(last_error().is_some());
        assert_eq!(ef_softmax_i8_plan_new(&params, ptr::null_mut()), -1);
        assert_eq!(errno(), libc::EINVAL);
        assert_eq!(ef_softmax_i8_plan_new(&params, &mut plan), 0);
        assert!(!plan.is_null());

        let mut data = [0i8; 4];
        let v = view(&mut data, I8, &[1, 4]);
        for (p, i, o) in [
            (ptr::null(), &v as *const _, &v as *const _),
            (plan as *const _, ptr::null(), &v as *const _),
            (plan as *const _, &v as *const _, ptr::null()),
        ] {
            assert_eq!(ef_softmax_i8_plan_run(p, i, o, ptr::null()), -1);
            assert_eq!(errno(), libc::EINVAL);
        }
        let mut no_shape = v;
        no_shape.shape = ptr::null();
        assert_eq!(ef_softmax_i8_plan_run(plan, &no_shape, &v, ptr::null()), -1);
        assert_eq!(errno(), libc::EINVAL);

        ef_softmax_i8_plan_free(plan);
        ef_softmax_i8_plan_free(ptr::null_mut());

        let mut lut: *mut EfLutI8Plan = ptr::null_mut();
        assert_eq!(ef_lut_i8_plan_new_table(I8, ptr::null(), &mut lut), -1);
        assert_eq!(errno(), libc::EINVAL);
        assert_eq!(
            ef_lut_i8_plan_new_sigmoid(I8, q(0.1, 0), q(1.0 / 256.0, -128), ptr::null_mut()),
            -1
        );
        assert_eq!(errno(), libc::EINVAL);
        ef_lut_i8_plan_free(ptr::null_mut());
    }
}

#[test]
fn bad_dtype_and_rank_return_einval() {
    let mut plan: *mut EfSoftmaxI8Plan = ptr::null_mut();
    let params = softmax_i8_params();
    let mut sig: *mut EfSigmoidPlan = ptr::null_mut();
    let sp = EfSigmoidParams {
        input_dtype: 99,
        input: q(0.1, 0),
        output_dtype: U8,
        output: q(1.0 / 256.0, 0),
    };
    // SAFETY: valid pointers; the values are what is under test.
    unsafe {
        assert_eq!(ef_sigmoid_plan_new(&sp, &mut sig), -1);
        assert_eq!(errno(), libc::EINVAL);
        assert!(sig.is_null());
        assert_eq!(ef_softmax_i8_plan_new(&params, &mut plan), 0);

        let mut data = [0i8; 8];
        let shape = [1usize; 7];
        let rank7 = view(&mut data, I8, &shape);
        assert_eq!(
            ef_softmax_i8_plan_run(plan, &rank7, &rank7, ptr::null()),
            -1
        );
        assert_eq!(errno(), libc::EINVAL);

        let bad = view(&mut data, 77, &[1, 8]);
        assert_eq!(ef_softmax_i8_plan_run(plan, &bad, &bad, ptr::null()), -1);
        assert_eq!(errno(), libc::EINVAL);
        ef_softmax_i8_plan_free(plan);
    }
}

#[test]
fn unsupported_output_quant_returns_enotsup_with_message() {
    let mut p = softmax_i8_params();
    p.output = q(1.0 / 256.0, 0);
    let mut plan: *mut EfSoftmaxI8Plan = ptr::null_mut();
    // SAFETY: valid pointers.
    let rc = unsafe { ef_softmax_i8_plan_new(&p, &mut plan) };
    assert_eq!(rc, -1);
    assert_eq!(errno(), libc::ENOTSUP);
    assert!(plan.is_null());
    let msg = last_error().expect("a message is stored");
    assert!(msg.contains("zero point"), "{msg}");
}

#[test]
fn invalid_params_return_einval() {
    let mut p = softmax_i8_params();
    p.input.scale = f32::NAN;
    let mut plan: *mut EfSoftmaxI8Plan = ptr::null_mut();
    // SAFETY: valid pointers.
    assert_eq!(unsafe { ef_softmax_i8_plan_new(&p, &mut plan) }, -1);
    assert_eq!(errno(), libc::EINVAL);
}

#[test]
fn panic_is_caught_as_internal_error() {
    assert_eq!(error::ffi(|| panic!("boom")), -1);
    assert_eq!(errno(), libc::EINVAL);
    assert_eq!(last_error().as_deref(), Some("internal error"));
}

#[test]
fn success_clears_the_last_error() {
    let _ = error::ffi(|| Err(error::CError::Invalid("x")));
    assert!(last_error().is_some());
    assert_eq!(error::ffi(|| Ok(())), 0);
    assert!(last_error().is_none());
}

struct Fixture {
    input: Vec<i8>,
    shape: Vec<usize>,
    input_quant: EfKernelQuant,
}

/// `testdata/kernels/<name>.safetensors`, under `EDGEFIRST_TESTDATA_DIR` when it is set.
fn fixture_path(name: &str) -> std::path::PathBuf {
    std::env::var_os("EDGEFIRST_TESTDATA_DIR")
        .map_or_else(
            || std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata"),
            std::path::PathBuf::from,
        )
        .join(format!("kernels/{name}.safetensors"))
}

fn load_yolo11() -> Fixture {
    let path = fixture_path("softmax_i8_yolo11_attn");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("fixture {}: {e}", path.display()));
    let (_, header) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
    let meta = header.metadata().as_ref().unwrap();
    let shape = meta["shape"]
        .trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .map(|d| d.trim().parse().unwrap())
        .collect();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let input = tensors
        .tensor("input")
        .unwrap()
        .data()
        .iter()
        .map(|&b| b as i8)
        .collect();
    Fixture {
        input,
        shape,
        input_quant: q(
            meta["input_scale"].parse().unwrap(),
            meta["input_zero_point"].parse().unwrap(),
        ),
    }
}

fn rust_softmax(f: &Fixture) -> Vec<i8> {
    let plan = k::SoftmaxI8Plan::new(k::SoftmaxI8Params {
        input: k::Quant::new(f.input_quant.scale, f.input_quant.zero_point),
        output: k::Quant::new(1.0 / 256.0, -128),
        beta: 1.0,
    })
    .unwrap();
    let mut out = vec![0i8; f.input.len()];
    plan.run(
        k::TensorRef::I8(k::View::contiguous(&f.input, &f.shape).unwrap()),
        k::TensorMut::I8(k::ViewMut::contiguous(&mut out, &f.shape).unwrap()),
        None,
    )
    .unwrap();
    out
}

fn c_softmax(f: &Fixture, pool: Option<&EfKernelPool>) -> Vec<i8> {
    let params = EfSoftmaxI8Params {
        input: f.input_quant,
        output: out_q(),
        beta: 1.0,
    };
    let mut plan: *mut EfSoftmaxI8Plan = ptr::null_mut();
    let mut input = f.input.clone();
    let mut out = vec![0i8; f.input.len()];
    let iv = view(&mut input, I8, &f.shape);
    let ov = view(&mut out, I8, &f.shape);
    // SAFETY: the views borrow buffers that outlive the calls.
    unsafe {
        assert_eq!(ef_softmax_i8_plan_new(&params, &mut plan), 0);
        let p = pool.map_or(ptr::null(), |p| p as *const _);
        assert_eq!(
            ef_softmax_i8_plan_run(plan, &iv, &ov, p),
            0,
            "{:?}",
            last_error()
        );
        ef_softmax_i8_plan_free(plan);
    }
    out
}

#[test]
fn softmax_i8_through_c_matches_rust_plan() {
    let f = load_yolo11();
    assert_eq!(c_softmax(&f, None), rust_softmax(&f));
}

unsafe extern "C" fn counting_run(
    ctx: *mut c_void,
    n_tasks: usize,
    task: unsafe extern "C" fn(*mut c_void, usize),
    arg: *mut c_void,
) {
    // SAFETY: ctx is the AtomicUsize owned by the test.
    unsafe { &*ctx.cast::<AtomicUsize>() }.fetch_add(1, Ordering::SeqCst);
    for i in 0..n_tasks {
        // SAFETY: task and arg are the pair the trampoline handed over.
        unsafe { task(arg, i) };
    }
}

#[test]
fn pool_callback_is_used() {
    let f = load_yolo11();
    let calls = AtomicUsize::new(0);
    let pool = EfKernelPool {
        ctx: ptr::from_ref(&calls).cast_mut().cast(),
        run: Some(counting_run),
    };
    assert_eq!(c_softmax(&f, Some(&pool)), rust_softmax(&f));
    assert!(
        calls.load(Ordering::SeqCst) > 0,
        "the C pool was never called"
    );
}

#[test]
fn pool_without_run_is_einval() {
    let pool = EfKernelPool {
        ctx: ptr::null_mut(),
        run: None,
    };
    let mut plan: *mut EfSoftmaxI8Plan = ptr::null_mut();
    let p = softmax_i8_params();
    let (mut a, mut b) = ([0i8; 4], [0i8; 4]);
    let (iv, ov) = (view(&mut a, I8, &[1, 4]), view(&mut b, I8, &[1, 4]));
    // SAFETY: valid pointers.
    unsafe {
        assert_eq!(ef_softmax_i8_plan_new(&p, &mut plan), 0);
        assert_eq!(ef_softmax_i8_plan_run(plan, &iv, &ov, &pool), -1);
        assert_eq!(errno(), libc::EINVAL);
        ef_softmax_i8_plan_free(plan);
    }
}

fn sigmoid_plan(dtype: u32, input: EfKernelQuant, output: EfKernelQuant) -> *mut EfSigmoidPlan {
    let p = EfSigmoidParams {
        input_dtype: dtype,
        input,
        output_dtype: dtype,
        output,
    };
    let mut plan: *mut EfSigmoidPlan = ptr::null_mut();
    // SAFETY: valid pointers.
    assert_eq!(unsafe { ef_sigmoid_plan_new(&p, &mut plan) }, 0);
    plan
}

#[test]
fn misaligned_data_returns_einval() {
    let f32_plan = sigmoid_plan(F32, q(1.0, 0), q(1.0, 0));
    let i16_plan = sigmoid_plan(I16, q(1.0 / 4096.0, 0), q(1.0 / 32768.0, 0));
    let mut src = [0u32; 8];
    let mut dst = [0u32; 8];
    let shape = [4usize];
    let at = |buf: &mut [u32], byte: usize, dtype: u32| EfKernelView {
        data: buf.as_mut_ptr().cast::<u8>().wrapping_add(byte).cast(),
        dtype,
        ndim: 1,
        shape: shape.as_ptr(),
        strides: ptr::null(),
    };
    // SAFETY: every view addresses memory inside its buffer; misaligned ones are refused first.
    unsafe {
        for (plan_run, dtype, odd) in [
            (
                &(|i: &EfKernelView, o: &EfKernelView| {
                    ef_sigmoid_plan_run(f32_plan, i, o, ptr::null())
                }) as &dyn Fn(&EfKernelView, &EfKernelView) -> i32,
                F32,
                2,
            ),
            (
                &|i: &EfKernelView, o: &EfKernelView| {
                    ef_sigmoid_plan_run(i16_plan, i, o, ptr::null())
                },
                I16,
                1,
            ),
        ] {
            assert_eq!(
                plan_run(&at(&mut src, 0, dtype), &at(&mut dst, 0, dtype)),
                0
            );
            assert_eq!(
                plan_run(&at(&mut src, odd, dtype), &at(&mut dst, 0, dtype)),
                -1
            );
            assert_eq!(errno(), libc::EINVAL);
            let msg = last_error().unwrap();
            assert!(msg.contains("not aligned"), "{msg}");
            assert_eq!(
                plan_run(&at(&mut src, 0, dtype), &at(&mut dst, odd, dtype)),
                -1
            );
            assert_eq!(errno(), libc::EINVAL);
        }
        ef_sigmoid_plan_free(f32_plan);
        ef_sigmoid_plan_free(i16_plan);
    }
}

#[test]
fn overlapping_views_return_einval_without_writing() {
    let params = softmax_i8_params();
    let mut plan: *mut EfSoftmaxI8Plan = ptr::null_mut();
    let mut buf = [5i8; 16];
    let base = buf.as_mut_ptr();
    let shape = [2usize, 4];
    let strides = [8isize, 2];
    let strided = |off: usize| EfKernelView {
        data: base.wrapping_add(off).cast(),
        dtype: I8,
        ndim: 2,
        shape: shape.as_ptr(),
        strides: strides.as_ptr(),
    };
    let dense = |off: usize| EfKernelView {
        data: base.wrapping_add(off).cast(),
        dtype: I8,
        ndim: 2,
        shape: shape.as_ptr(),
        strides: ptr::null(),
    };
    // SAFETY: every view addresses elements of `buf`.
    unsafe {
        assert_eq!(ef_softmax_i8_plan_new(&params, &mut plan), 0);
        for (i, o) in [
            (dense(0), dense(0)),
            (dense(0), dense(4)),
            (strided(0), strided(1)),
            (strided(1), strided(0)),
        ] {
            assert_eq!(ef_softmax_i8_plan_run(plan, &i, &o, ptr::null()), -1);
            assert_eq!(errno(), libc::EINVAL);
            assert_eq!(
                last_error().as_deref(),
                Some("input and output views overlap")
            );
        }
        assert_eq!(buf, [5; 16], "a rejected run wrote");
        assert_eq!(
            ef_softmax_i8_plan_run(plan, &dense(0), &dense(8), ptr::null()),
            0
        );
        assert_eq!(buf[8..], [-64; 8]);
        ef_softmax_i8_plan_free(plan);
    }
}

/// Pre-sets `*out` to a dangling non-NULL value, runs `new`, and checks it fails with EINVAL
/// and leaves NULL.
fn assert_fails_with_null_out<P>(what: &str, new: impl FnOnce(*mut *mut P) -> i32) {
    assert_new_fails(what, libc::EINVAL, new);
}

/// Pre-sets `*out` to a dangling non-NULL value, runs `new`, and checks it fails with `want`
/// and leaves NULL.
fn assert_new_fails<P>(what: &str, want: i32, new: impl FnOnce(*mut *mut P) -> i32) {
    let mut out: *mut P = ptr::NonNull::dangling().as_ptr();
    assert_eq!(new(&mut out), -1, "{what}");
    assert_eq!(errno(), want, "{what}: {:?}", last_error());
    assert!(out.is_null(), "{what}: *out was not cleared");
}

#[test]
fn every_plan_new_failure_leaves_out_null() {
    let good_q = q(0.1, 0);
    // SAFETY: NULL params and bad values are the point; `out` is always writable.
    unsafe {
        assert_fails_with_null_out("softmax_i8 NULL", |o| {
            ef_softmax_i8_plan_new(ptr::null(), o)
        });
        assert_fails_with_null_out("softmax_i16 NULL", |o| {
            ef_softmax_i16_plan_new(ptr::null(), o)
        });
        assert_fails_with_null_out("sigmoid NULL", |o| ef_sigmoid_plan_new(ptr::null(), o));
        assert_fails_with_null_out("dfl NULL", |o| ef_dfl_plan_new(ptr::null(), o));
        assert_fails_with_null_out("dist2bbox NULL", |o| ef_dist2bbox_plan_new(ptr::null(), o));
        assert_fails_with_null_out("dfl_box NULL", |o| ef_dfl_box_plan_new(ptr::null(), o));
        let mut p = softmax_i8_params();
        p.input.scale = f32::NAN;
        assert_fails_with_null_out("softmax_i8 bad scale", |o| ef_softmax_i8_plan_new(&p, o));
        let table = [0u8; 256];
        assert_fails_with_null_out("lut NULL table", |o| {
            ef_lut_i8_plan_new_table(I8, ptr::null(), o)
        });
        assert_fails_with_null_out("lut table bad dtype", |o| {
            ef_lut_i8_plan_new_table(99, table.as_ptr(), o)
        });
        assert_fails_with_null_out("lut table i16", |o| {
            ef_lut_i8_plan_new_table(I16, table.as_ptr(), o)
        });
        assert_fails_with_null_out("lut sigmoid bad dtype", |o| {
            ef_lut_i8_plan_new_sigmoid(99, good_q, out_q(), o)
        });
        assert_fails_with_null_out("lut sigmoid bad scale", |o| {
            ef_lut_i8_plan_new_sigmoid(I8, q(0.0, 0), out_q(), o)
        });
    }
}

unsafe extern "C" fn serial_run(
    _ctx: *mut c_void,
    n_tasks: usize,
    task: unsafe extern "C" fn(*mut c_void, usize),
    arg: *mut c_void,
) {
    for i in 0..n_tasks {
        // SAFETY: task and arg are the pair the trampoline handed over.
        unsafe { task(arg, i) };
    }
}

fn serial_pool() -> EfKernelPool {
    EfKernelPool {
        ctx: ptr::null_mut(),
        run: Some(serial_run),
    }
}

#[test]
fn a_panicking_pool_task_sets_the_flag() {
    let Ok(pool) = pool::CPool::new(&serial_pool()) else {
        panic!("a pool with a run callback is valid");
    };
    let ran = AtomicUsize::new(0);
    pool.run(3, &|_| {
        ran.fetch_add(1, Ordering::SeqCst);
    });
    assert!(!pool.panicked());
    pool.run(3, &|i| {
        ran.fetch_add(1, Ordering::SeqCst);
        if i == 1 {
            panic!("task panic");
        }
    });
    assert_eq!(ran.load(Ordering::SeqCst), 6, "the other tasks still ran");
    assert!(pool.panicked());
}

#[test]
fn run_plan_maps_a_pool_task_panic_to_internal_error() {
    let c_pool = serial_pool();
    let (mut a, mut b) = ([0i8; 4], [0i8; 4]);
    let (iv, ov) = (view(&mut a, I8, &[1, 4]), view(&mut b, I8, &[1, 4]));
    let plan = 0u8;
    let rc = error::ffi(|| {
        // SAFETY: valid pointers; the closure stands in for a plan whose task panics.
        unsafe {
            plans::run_plan(&plan, &iv, &ov, &c_pool, |_, _, _, pool| {
                if let Some(pool) = pool {
                    pool.run(2, &|i| assert_ne!(i, 1, "task panic"));
                }
                Ok(())
            })
        }
    });
    assert_eq!(rc, -1);
    assert_eq!(errno(), libc::EINVAL);
    assert_eq!(last_error().as_deref(), Some("internal error"));
}

const OPT_NEON: u32 = EfTfliteKernels::NeonRuy as u32;
const REFERENCE: u32 = EfTfliteKernels::Reference as u32;
const ACT_NONE: u32 = EfKernelActivation::None as u32;

/// A multi-tensor fixture from `testdata/kernels`: tensors as raw bytes with shapes, and metadata.
struct Graph {
    meta: std::collections::HashMap<String, String>,
    tensors: std::collections::HashMap<String, (Vec<usize>, Vec<u8>)>,
}

impl Graph {
    fn load(name: &str) -> Self {
        let path = fixture_path(name);
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let (_, header) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
        let meta = header.metadata().clone().unwrap().into_iter().collect();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let tensors = st
            .tensors()
            .into_iter()
            .map(|(k, v)| (k, (v.shape().to_vec(), v.data().to_vec())))
            .collect();
        Self { meta, tensors }
    }

    fn quant(&self, role: &str) -> EfKernelQuant {
        q(
            self.meta[&format!("{role}_scale")].parse().unwrap(),
            self.meta[&format!("{role}_zero_point")].parse().unwrap(),
        )
    }

    fn shape(&self, key: &str) -> Vec<usize> {
        self.tensors[key].0.clone()
    }

    fn i8(&self, key: &str) -> Vec<i8> {
        self.tensors[key].1.iter().map(|&b| b as i8).collect()
    }

    fn i32(&self, key: &str) -> Vec<i32> {
        self.tensors[key]
            .1
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn f32_list(&self, key: &str) -> Vec<f32> {
        self.meta[key]
            .trim_matches(|c| c == '[' || c == ']')
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect()
    }
}

fn strided_view<T>(data: &mut [T], dtype: u32, shape: &[usize], strides: &[isize]) -> EfKernelView {
    EfKernelView {
        data: data.as_mut_ptr().cast(),
        dtype,
        ndim: shape.len(),
        shape: shape.as_ptr(),
        strides: strides.as_ptr(),
    }
}

fn kq(x: EfKernelQuant) -> k::Quant {
    x.into()
}

fn binary_params(g: &Graph, op: u32, kernels: u32) -> EfBinaryI8Params {
    let activation = match g.meta["activation"].as_str() {
        "none" => EfKernelActivation::None,
        "relu" => EfKernelActivation::Relu,
        "relu6" => EfKernelActivation::Relu6,
        other => panic!("activation {other}"),
    } as u32;
    EfBinaryI8Params {
        op,
        lhs: g.quant("lhs"),
        rhs: g.quant("rhs"),
        output: g.quant("output"),
        activation,
        kernels,
    }
}

fn new_binary(p: &EfBinaryI8Params) -> *mut EfBinaryI8Plan {
    let mut plan = ptr::null_mut();
    // SAFETY: valid pointers.
    assert_eq!(
        unsafe { ef_binary_i8_plan_new(p, &mut plan) },
        0,
        "{:?}",
        last_error()
    );
    plan
}

#[test]
fn binary_through_c_matches_the_tflite_fixtures() {
    for (name, op) in [
        ("add_i8_odd", EfBinaryOp::Add),
        ("sub_i8_odd", EfBinaryOp::Sub),
        ("mul_i8_odd", EfBinaryOp::Mul),
        ("mul_i8_scalar", EfBinaryOp::Mul),
    ] {
        let g = Graph::load(name);
        let shape = g.shape("lhs");
        let (mut lhs, mut rhs) = (g.i8("lhs"), g.i8("rhs"));
        let mut out = vec![0x55i8; lhs.len()];
        let zeros = vec![0isize; shape.len()];
        let lv = view(&mut lhs, I8, &shape);
        // A broadcast scalar is a view of the lhs shape with zero strides.
        let rv = if rhs.len() == 1 {
            strided_view(&mut rhs, I8, &shape, &zeros)
        } else {
            view(&mut rhs, I8, &shape)
        };
        let ov = view(&mut out, I8, &shape);
        let plan = new_binary(&binary_params(&g, op as u32, REFERENCE));
        // SAFETY: the views borrow buffers that outlive the calls.
        unsafe {
            assert_eq!(
                ef_binary_i8_plan_run(plan, &lv, &rv, &ov, ptr::null()),
                0,
                "{name}: {:?}",
                last_error()
            );
            ef_binary_i8_plan_free(plan);
        }
        assert_eq!(out, g.i8("output"), "{name}");
    }
}

#[test]
fn binary_rhs_may_share_the_lhs_buffer_but_not_the_output() {
    let p = EfBinaryI8Params {
        op: EfBinaryOp::Add as u32,
        lhs: q(0.5, 0),
        rhs: q(0.5, 0),
        output: q(1.0, 0),
        activation: ACT_NONE,
        kernels: OPT_NEON,
    };
    let plan = new_binary(&p);
    let mut buf = [2i8, 4, -6, 8, 5, 5, 5, 5];
    let base = buf.as_mut_ptr();
    let shape = [4usize];
    let at = |off: usize| EfKernelView {
        data: base.wrapping_add(off).cast(),
        dtype: I8,
        ndim: 1,
        shape: shape.as_ptr(),
        strides: ptr::null(),
    };
    // SAFETY: every view addresses elements of `buf`.
    unsafe {
        // The output overlapping only the rhs is refused before anything is written.
        assert_eq!(
            ef_binary_i8_plan_run(plan, &at(0), &at(2), &at(4), ptr::null()),
            -1
        );
        assert_eq!(errno(), libc::EINVAL);
        assert_eq!(
            last_error().as_deref(),
            Some("input and output views overlap")
        );
        assert_eq!(buf[4..], [5; 4]);
        // Both inputs on the same bytes: x + x.
        assert_eq!(
            ef_binary_i8_plan_run(plan, &at(0), &at(0), &at(4), ptr::null()),
            0
        );
        assert_eq!(buf[4..], [2, 4, -6, 8]);
        assert_eq!(
            ef_binary_i8_plan_run(plan, ptr::null(), &at(0), &at(4), ptr::null()),
            -1
        );
        assert_eq!(errno(), libc::EINVAL);
        assert_eq!(
            ef_binary_i8_plan_run(plan, &at(0), ptr::null(), &at(4), ptr::null()),
            -1
        );
        assert_eq!(errno(), libc::EINVAL);
        ef_binary_i8_plan_free(plan);
        ef_binary_i8_plan_free(ptr::null_mut());
    }
}

#[test]
fn binary_table_for_rhs_follows_the_rust_plan() {
    let p = EfBinaryI8Params {
        op: EfBinaryOp::Mul as u32,
        lhs: q(0.0579, -128),
        rhs: q(0.0391, 5),
        output: q(0.0015, -100),
        activation: ACT_NONE,
        kernels: OPT_NEON,
    };
    let rust = k::BinaryI8Plan::new(k::BinaryI8Params {
        op: k::BinaryOp::Mul,
        lhs: kq(p.lhs),
        rhs: kq(p.rhs),
        output: kq(p.output),
        activation: k::Activation::None,
        kernels: k::TfliteKernels::NeonRuy,
    })
    .unwrap();
    let split = (-128i8..=127)
        .find(|&r| rust.table_for_rhs(r, 17).is_none())
        .expect("some rhs gives different NEON-block and scalar tables");
    let plan = new_binary(&p);
    let mut table = [0u8; 256];
    // SAFETY: valid pointers; `table` holds 256 bytes.
    unsafe {
        assert_eq!(
            ef_binary_i8_plan_table_for_rhs(plan, split, 17, table.as_mut_ptr()),
            -1
        );
        assert_eq!(errno(), libc::ENOTSUP);
        assert!(last_error().is_some());
        for len in [16usize, 17, 0] {
            let rhs = if len == 17 { 3 } else { split };
            match rust.table_for_rhs(rhs, len) {
                Some(want) => {
                    assert_eq!(
                        ef_binary_i8_plan_table_for_rhs(plan, rhs, len, table.as_mut_ptr()),
                        0
                    );
                    assert_eq!(table, want, "rhs {rhs} len {len}");
                }
                None => assert_eq!(
                    ef_binary_i8_plan_table_for_rhs(plan, rhs, len, table.as_mut_ptr()),
                    -1
                ),
            }
        }
        assert_eq!(
            ef_binary_i8_plan_table_for_rhs(plan, 0, 16, ptr::null_mut()),
            -1
        );
        assert_eq!(errno(), libc::EINVAL);
        assert_eq!(
            ef_binary_i8_plan_table_for_rhs(ptr::null(), 0, 16, table.as_mut_ptr()),
            -1
        );
        assert_eq!(errno(), libc::EINVAL);
        ef_binary_i8_plan_free(plan);
    }
}

fn new_matmul(p: &EfMatmulI8Params) -> *mut EfMatmulI8Plan {
    let mut plan = ptr::null_mut();
    // SAFETY: valid pointers, arrays included.
    assert_eq!(
        unsafe { ef_matmul_i8_plan_new(p, &mut plan) },
        0,
        "{:?}",
        last_error()
    );
    plan
}

fn requant_bmm(l: f32, r: f32, o: f32) -> EfKernelRequant {
    let mut out = EfKernelRequant {
        multiplier: 0,
        shift: 0,
    };
    // SAFETY: `out` is writable.
    assert_eq!(
        unsafe { ef_kernel_requant_tflite_batch_matmul(l, r, o, &mut out) },
        0
    );
    out
}

#[test]
fn requant_helpers_equal_the_rust_ones() {
    let r = requant_bmm(0.0412, 0.0377, 0.71);
    assert_eq!(
        k::Requant::from(r),
        k::Requant::tflite_batch_matmul(0.0412, 0.0377, 0.71).unwrap()
    );
    let mut c = EfKernelRequant {
        multiplier: 0,
        shift: 0,
    };
    // SAFETY: `c` is writable; NULL and bad values are the point.
    unsafe {
        assert_eq!(ef_kernel_requant_tflite_conv(0.05, 0.01, 0.04, &mut c), 0);
        assert_eq!(
            k::Requant::from(c),
            k::Requant::tflite_conv(0.05, 0.01, 0.04).unwrap()
        );
        assert_eq!(
            ef_kernel_requant_tflite_conv(0.05, f32::NAN, 0.04, &mut c),
            -1
        );
        assert_eq!(errno(), libc::EINVAL);
        assert_eq!(
            ef_kernel_requant_tflite_batch_matmul(0.0, 1.0, 1.0, &mut c),
            -1
        );
        assert_eq!(errno(), libc::EINVAL);
        assert_eq!(
            ef_kernel_requant_tflite_batch_matmul(1.0, 1.0, 1.0, ptr::null_mut()),
            -1
        );
        assert_eq!(errno(), libc::EINVAL);
    }
}

#[test]
fn batch_matmul_with_adjoint_views_matches_the_tflite_fixture() {
    let g = Graph::load("batch_matmul_i8_adj");
    let (ls, rs) = (g.shape("lhs"), g.shape("rhs"));
    // Stored as [b, k, m] and [b, n, k]; read as [b, m, k] and [b, k, n].
    let (b, k_, m, n) = (ls[0], ls[1], ls[2], rs[1]);
    let (mut lhs, mut rhs) = (g.i8("lhs"), g.i8("rhs"));
    let (lshape, lstr) = ([b, m, k_], [(k_ * m) as isize, 1, m as isize]);
    let (rshape, rstr) = ([b, k_, n], [(n * k_) as isize, 1, k_ as isize]);
    let oshape = [b, m, n];
    let (lq, rq, oq) = (g.quant("lhs"), g.quant("rhs"), g.quant("output"));
    let requant = [requant_bmm(lq.scale, rq.scale, oq.scale)];
    let p = EfMatmulI8Params {
        lhs_zero_point: lq.zero_point,
        rhs_zero_point: rq.zero_point,
        output: oq,
        requant: requant.as_ptr(),
        n_requant: 1,
        bias: ptr::null(),
        n_bias: 0,
        activation: ACT_NONE,
        kernels: REFERENCE,
    };
    let plan = new_matmul(&p);
    let mut out = vec![0x55i8; b * m * n];
    let calls = AtomicUsize::new(0);
    let pool = EfKernelPool {
        ctx: ptr::from_ref(&calls).cast_mut().cast(),
        run: Some(counting_run),
    };
    // SAFETY: the views borrow buffers that outlive the calls.
    unsafe {
        let (lv, rv) = (
            strided_view(&mut lhs, I8, &lshape, &lstr),
            strided_view(&mut rhs, I8, &rshape, &rstr),
        );
        let ov = view(&mut out, I8, &oshape);
        assert_eq!(
            ef_matmul_i8_plan_run(plan, &lv, &rv, &ov, &pool),
            0,
            "{:?}",
            last_error()
        );
        ef_matmul_i8_plan_free(plan);
    }
    assert!(
        calls.load(Ordering::SeqCst) > 0,
        "the C pool was never called"
    );
    assert_eq!(out, g.i8("output"));
}

#[test]
fn conv_1x1_matches_the_tflite_fixture() {
    let g = Graph::load("conv1x1_i8");
    let ishape = g.shape("input");
    let fshape = g.shape("filter");
    let (cin, cout) = (ishape[3], fshape[0]);
    let pixels = ishape[0] * ishape[1] * ishape[2];
    let (iq, oq) = (g.quant("input"), g.quant("output"));
    let requant: Vec<EfKernelRequant> = g
        .f32_list("filter_scales")
        .iter()
        .map(|&s| {
            let mut r = EfKernelRequant {
                multiplier: 0,
                shift: 0,
            };
            // SAFETY: `r` is writable.
            assert_eq!(
                unsafe { ef_kernel_requant_tflite_conv(iq.scale, s, oq.scale, &mut r) },
                0
            );
            r
        })
        .collect();
    let bias = g.i32("bias");
    assert_eq!(g.meta["activation"], "relu6");
    let p = EfMatmulI8Params {
        lhs_zero_point: iq.zero_point,
        rhs_zero_point: 0,
        output: oq,
        requant: requant.as_ptr(),
        n_requant: requant.len(),
        bias: bias.as_ptr(),
        n_bias: bias.len(),
        activation: EfKernelActivation::Relu6 as u32,
        kernels: REFERENCE,
    };
    let plan = new_matmul(&p);
    let (mut input, mut filter) = (g.i8("input"), g.i8("filter"));
    let mut out = vec![0x55i8; pixels * cout];
    let (lshape, rshape, oshape) = ([pixels, cin], [cin, cout], [pixels, cout]);
    let rstr = [1isize, cin as isize];
    // SAFETY: the views borrow buffers that outlive the calls.
    unsafe {
        let lv = view(&mut input, I8, &lshape);
        let rv = strided_view(&mut filter, I8, &rshape, &rstr);
        let ov = view(&mut out, I8, &oshape);
        assert_eq!(
            ef_matmul_i8_plan_run(plan, &lv, &rv, &ov, ptr::null()),
            0,
            "{:?}",
            last_error()
        );
        // One requantization per column is checked against the call's columns.
        let narrow = [pixels, cout - 1];
        let mut small = vec![0x55i8; pixels * (cout - 1)];
        let rv = strided_view(&mut filter, I8, &[cin, cout - 1], &rstr);
        let sv = view(&mut small, I8, &narrow);
        assert_eq!(ef_matmul_i8_plan_run(plan, &lv, &rv, &sv, ptr::null()), -1);
        assert_eq!(errno(), libc::EINVAL);
        assert!(small.iter().all(|&b| b == 0x55));
        ef_matmul_i8_plan_free(plan);
    }
    assert_eq!(out, g.i8("output"));
}

#[test]
fn matmul_params_arrays_and_enums_are_checked() {
    let requant = [requant_bmm(0.5, 0.5, 1.0)];
    let good = EfMatmulI8Params {
        lhs_zero_point: 0,
        rhs_zero_point: 0,
        output: q(1.0, 0),
        requant: requant.as_ptr(),
        n_requant: 1,
        bias: ptr::null(),
        n_bias: 0,
        activation: ACT_NONE,
        kernels: OPT_NEON,
    };
    let fails = |p: EfMatmulI8Params, want: i32, what: &str| {
        let mut out: *mut EfMatmulI8Plan = ptr::NonNull::dangling().as_ptr();
        // SAFETY: valid pointers; the values are under test.
        assert_eq!(unsafe { ef_matmul_i8_plan_new(&p, &mut out) }, -1, "{what}");
        assert_eq!(errno(), want, "{what}");
        assert!(out.is_null(), "{what}");
    };
    fails(
        EfMatmulI8Params {
            requant: ptr::null(),
            ..good
        },
        libc::EINVAL,
        "NULL requant",
    );
    fails(
        EfMatmulI8Params { n_bias: 2, ..good },
        libc::EINVAL,
        "NULL bias",
    );
    fails(
        EfMatmulI8Params {
            n_requant: 0,
            ..good
        },
        libc::EINVAL,
        "no requant",
    );
    fails(
        EfMatmulI8Params { kernels: 4, ..good },
        libc::EINVAL,
        "kernels",
    );
    fails(
        EfMatmulI8Params {
            activation: 4,
            ..good
        },
        libc::EINVAL,
        "activation",
    );
    // Ruy rounding takes multipliers below 0.5 only.
    let big = [EfKernelRequant {
        multiplier: 1 << 30,
        shift: 1,
    }];
    fails(
        EfMatmulI8Params {
            requant: big.as_ptr(),
            ..good
        },
        libc::ENOTSUP,
        "ruy multiplier",
    );
    let plan = new_matmul(&EfMatmulI8Params {
        requant: big.as_ptr(),
        kernels: REFERENCE,
        ..good
    });
    // SAFETY: a live plan.
    unsafe { ef_matmul_i8_plan_free(plan) };
    assert_eq!(ef_matmul_i8_check_depth(4096), 0);
    assert_eq!(ef_matmul_i8_check_depth(4097), -1);
    assert_eq!(errno(), libc::ENOTSUP);
}

fn rust_softmax_dot(g: &Graph) -> k::SoftmaxDotI8Plan {
    let probs = g.quant("probs");
    k::SoftmaxDotI8Plan::new(k::SoftmaxDotI8Params {
        softmax: k::SoftmaxI8Params {
            input: kq(g.quant("input")),
            output: kq(probs),
            beta: 1.0,
        },
        weights: g.i8("filter"),
        projection: k::MatMulI8Params::tflite_conv(
            kq(probs),
            &g.f32_list("filter_scales"),
            kq(g.quant("output")),
            vec![0],
            k::Activation::None,
            k::TfliteKernels::Reference,
        )
        .unwrap(),
    })
    .unwrap()
}

#[test]
fn softmax_dot_through_c_matches_rust_plan() {
    let g = Graph::load("dfl_head_i8");
    let (iq, probs, oq) = (g.quant("input"), g.quant("probs"), g.quant("output"));
    let weights = g.i8("filter");
    let fs = g.f32_list("filter_scales");
    let mut r = EfKernelRequant {
        multiplier: 0,
        shift: 0,
    };
    // SAFETY: `r` is writable.
    assert_eq!(
        unsafe { ef_kernel_requant_tflite_conv(probs.scale, fs[0], oq.scale, &mut r) },
        0
    );
    let bias = [0i32];
    let p = EfSoftmaxDotI8Params {
        softmax: EfSoftmaxI8Params {
            input: iq,
            output: probs,
            beta: 1.0,
        },
        weights: weights.as_ptr(),
        n_weights: weights.len(),
        projection: EfMatmulI8Params {
            lhs_zero_point: probs.zero_point,
            rhs_zero_point: 0,
            output: oq,
            requant: &r,
            n_requant: 1,
            bias: bias.as_ptr(),
            n_bias: 1,
            activation: ACT_NONE,
            kernels: REFERENCE,
        },
    };
    let mut plan = ptr::null_mut();
    let shape = g.shape("input");
    let oshape = [shape[0], shape[1], shape[2], 1];
    let mut input = g.i8("input");
    let mut out = vec![0x55i8; oshape.iter().product()];
    // SAFETY: valid pointers; the views borrow buffers that outlive the calls.
    unsafe {
        assert_eq!(
            ef_softmax_dot_i8_plan_new(&p, &mut plan),
            0,
            "{:?}",
            last_error()
        );
        let iv = view(&mut input, I8, &shape);
        let ov = view(&mut out, I8, &oshape);
        assert_eq!(
            ef_softmax_dot_i8_plan_run(plan, &iv, &ov, &serial_pool()),
            0,
            "{:?}",
            last_error()
        );
        // The row width must match the weights.
        let bad = [shape[0], shape[1], shape[2] * 2, 8];
        let bv = view(&mut input, I8, &bad);
        let mut small = vec![0x55i8; shape[1] * shape[2] * 2];
        let sv = view(&mut small, I8, &[shape[0], shape[1], shape[2] * 2, 1]);
        assert_eq!(ef_softmax_dot_i8_plan_run(plan, &bv, &sv, ptr::null()), -1);
        assert_eq!(errno(), libc::EINVAL);
        assert!(small.iter().all(|&b| b == 0x55));
        ef_softmax_dot_i8_plan_free(plan);
        ef_softmax_dot_i8_plan_free(ptr::null_mut());
    }
    let mut want = vec![0i8; out.len()];
    let input = g.i8("input");
    rust_softmax_dot(&g)
        .run(
            k::TensorRef::I8(k::View::contiguous(&input, &shape).unwrap()),
            k::TensorMut::I8(k::ViewMut::contiguous(&mut want, &oshape).unwrap()),
            None,
        )
        .unwrap();
    assert_eq!(out, want);

    // The nested arrays are checked like matmul's.
    let mut bad = p;
    bad.weights = ptr::null();
    assert_fails_with_null_out("softmax_dot NULL weights", |o| unsafe {
        ef_softmax_dot_i8_plan_new(&bad, o)
    });
    let mut bad = p;
    bad.projection.lhs_zero_point = 0;
    assert_fails_with_null_out("softmax_dot projection zero point", |o| unsafe {
        ef_softmax_dot_i8_plan_new(&bad, o)
    });
    let mut bad = p;
    bad.projection.kernels = 7;
    assert_fails_with_null_out("softmax_dot kernels", |o| unsafe {
        ef_softmax_dot_i8_plan_new(&bad, o)
    });
}

/// The attention fixture's params, with the Q `MUL` table from the C binary plan.
fn attention_params(g: &Graph, q_mul: &mut EfAttentionQMul) -> EfAttentionI8Params {
    let (h, n, dk) = {
        let s = g.shape("q");
        (s[1], s[2], s[3])
    };
    let mul = new_binary(&EfBinaryI8Params {
        op: EfBinaryOp::Mul as u32,
        lhs: g.quant("q"),
        rhs: g.quant("mul_rhs"),
        output: g.quant("q_scaled"),
        activation: ACT_NONE,
        kernels: REFERENCE,
    });
    let rhs: i8 = g.meta["mul_rhs_value"].parse().unwrap();
    // SAFETY: a live plan and a 256-byte table.
    unsafe {
        assert_eq!(
            ef_binary_i8_plan_table_for_rhs(mul, rhs, h * n * dk, q_mul.table.as_mut_ptr()),
            0
        );
        ef_binary_i8_plan_free(mul);
    }
    q_mul.output = g.quant("q_scaled");
    EfAttentionI8Params {
        q: g.quant("q"),
        q_mul,
        k: g.quant("k"),
        v: g.quant("v"),
        scores: g.quant("scores"),
        beta: 1.0,
        probs: g.quant("probs"),
        output: g.quant("output"),
        kernels: REFERENCE,
    }
}

fn rust_attention(p: &EfAttentionI8Params) -> k::AttentionI8Plan {
    // SAFETY: `q_mul` points at the caller's live table.
    let m = unsafe { &*p.q_mul };
    k::AttentionI8Plan::new(k::AttentionI8Params {
        q: kq(p.q),
        q_mul: Some(k::AttentionQMul {
            table: m.table,
            output: kq(m.output),
        }),
        k: kq(p.k),
        v: kq(p.v),
        scores: kq(p.scores),
        beta: p.beta,
        probs: kq(p.probs),
        output: kq(p.output),
        kernels: k::TfliteKernels::Reference,
    })
    .unwrap()
}

#[test]
fn attention_through_c_matches_rust_plan_and_reuses_its_scratch() {
    let g = Graph::load("attention_i8_odd");
    let mut q_mul = EfAttentionQMul {
        table: [0; 256],
        output: q(1.0, 0),
    };
    let p = attention_params(&g, &mut q_mul);
    let s = g.shape("v");
    let (h, n, dk, dv) = (s[1], s[2], g.shape("q")[3], s[3]);
    let (qs, vs, os) = ([h, n, dk], [h, n, dv], [h, n, dv]);
    let (mut qd, mut kd, mut vd) = (g.i8("q"), g.i8("k"), g.i8("v"));
    let rust = rust_attention(&p);
    let rust_run = |q: &[i8]| {
        let mut out = vec![0i8; h * n * dv];
        let mut scratch = rust.scratch(h, n, n, dk, dv).unwrap();
        rust.run(
            k::TensorRef::I8(k::View::contiguous(q, &qs).unwrap()),
            k::TensorRef::I8(k::View::contiguous(&g.i8("k"), &qs).unwrap()),
            k::TensorRef::I8(k::View::contiguous(&g.i8("v"), &vs).unwrap()),
            k::TensorMut::I8(k::ViewMut::contiguous(&mut out, &os).unwrap()),
            &mut scratch,
            None,
        )
        .unwrap();
        out
    };
    let mut plan = ptr::null_mut();
    let mut scratch = ptr::null_mut();
    let mut out = vec![0x55i8; h * n * dv];
    // SAFETY: valid pointers; the views borrow buffers that outlive the calls.
    unsafe {
        assert_eq!(
            ef_attention_i8_plan_new(&p, &mut plan),
            0,
            "{:?}",
            last_error()
        );
        assert_eq!(
            ef_attention_scratch_new(plan, h, n, n, dk, dv, &mut scratch),
            0
        );
        let (kv, vv) = (view(&mut kd, I8, &qs), view(&mut vd, I8, &vs));
        let ov = view(&mut out, I8, &os);
        let qv = view(&mut qd, I8, &qs);
        assert_eq!(
            ef_attention_i8_plan_run(plan, &qv, &kv, &vv, &ov, scratch, &serial_pool()),
            0,
            "{:?}",
            last_error()
        );
        assert_eq!(out, rust_run(&g.i8("q")));
        if rust.tier() == k::Tier::Scalar {
            assert_eq!(out, g.i8("output"), "scalar tier vs the TFLite chain");
        }
        // The same scratch serves a second call with other queries.
        let q2: Vec<i8> = g.i8("q").iter().map(|&x| x.wrapping_mul(3) / 2).collect();
        let mut q2c = q2.clone();
        let q2v = view(&mut q2c, I8, &qs);
        assert_eq!(
            ef_attention_i8_plan_run(plan, &q2v, &kv, &vv, &ov, scratch, ptr::null()),
            0
        );
        assert_eq!(out, rust_run(&q2));

        // A scratch for other shapes is refused before anything is written.
        let mut other = ptr::null_mut();
        assert_eq!(
            ef_attention_scratch_new(plan, h, n, n - 1, dk, dv, &mut other),
            0
        );
        out.fill(0x55);
        assert_eq!(
            ef_attention_i8_plan_run(plan, &qv, &kv, &vv, &ov, other, ptr::null()),
            -1
        );
        assert_eq!(errno(), libc::EINVAL);
        assert!(out.iter().all(|&b| b == 0x55));
        ef_attention_scratch_free(other);
        assert_eq!(
            ef_attention_i8_plan_run(plan, &qv, &kv, &vv, &ov, ptr::null_mut(), ptr::null()),
            -1
        );
        assert_eq!(errno(), libc::EINVAL);
        // The output overlapping any one input is refused.
        let into_v = view(&mut vd, I8, &os);
        assert_eq!(
            ef_attention_i8_plan_run(plan, &qv, &kv, &vv, &into_v, scratch, ptr::null()),
            -1
        );
        assert_eq!(errno(), libc::EINVAL);
        assert_eq!(vd, g.i8("v"));
        ef_attention_scratch_free(scratch);
        ef_attention_scratch_free(ptr::null_mut());
        ef_attention_i8_plan_free(plan);
        ef_attention_i8_plan_free(ptr::null_mut());
    }
}

#[test]
fn attention_reads_head_slices_of_one_buffer() {
    let g = Graph::load("attention_i8_odd");
    let mut q_mul = EfAttentionQMul {
        table: [0; 256],
        output: q(1.0, 0),
    };
    let p = attention_params(&g, &mut q_mul);
    let (h, n, dk, dv) = {
        let (a, b) = (g.shape("q"), g.shape("v"));
        (a[1], a[2], a[3], b[3])
    };
    // Token-major [n, h, dk + dk + dv], as a fused QKV projection writes it.
    let c = 2 * dk + dv;
    let (qd, kd, vd) = (g.i8("q"), g.i8("k"), g.i8("v"));
    let mut qkv = vec![0i8; n * h * c];
    for t in 0..n {
        for hh in 0..h {
            let at = (t * h + hh) * c;
            let src = |d: &[i8], w: usize| d[(hh * n + t) * w..(hh * n + t + 1) * w].to_vec();
            qkv[at..at + dk].copy_from_slice(&src(&qd, dk));
            qkv[at + dk..at + 2 * dk].copy_from_slice(&src(&kd, dk));
            qkv[at + 2 * dk..at + c].copy_from_slice(&src(&vd, dv));
        }
    }
    let st = [c as isize, (h * c) as isize, 1];
    let (qs, vs) = ([h, n, dk], [h, n, dv]);
    let base = qkv.as_mut_ptr();
    let slice = |off: usize, shape: &[usize; 3]| EfKernelView {
        data: base.wrapping_add(off).cast(),
        dtype: I8,
        ndim: 3,
        shape: shape.as_ptr(),
        strides: st.as_ptr(),
    };
    let mut out = vec![0x55i8; h * n * dv];
    let mut want = vec![0i8; h * n * dv];
    let rust = rust_attention(&p);
    let mut rs = rust.scratch(h, n, n, dk, dv).unwrap();
    rust.run(
        k::TensorRef::I8(k::View::contiguous(&qd, &qs).unwrap()),
        k::TensorRef::I8(k::View::contiguous(&kd, &qs).unwrap()),
        k::TensorRef::I8(k::View::contiguous(&vd, &vs).unwrap()),
        k::TensorMut::I8(k::ViewMut::contiguous(&mut want, &vs).unwrap()),
        &mut rs,
        None,
    )
    .unwrap();
    let (mut plan, mut scratch) = (ptr::null_mut(), ptr::null_mut());
    // SAFETY: every view addresses elements of `qkv` or `out`, which outlive the calls.
    unsafe {
        assert_eq!(ef_attention_i8_plan_new(&p, &mut plan), 0);
        assert_eq!(
            ef_attention_scratch_new(plan, h, n, n, dk, dv, &mut scratch),
            0
        );
        let ov = view(&mut out, I8, &vs);
        assert_eq!(
            ef_attention_i8_plan_run(
                plan,
                &slice(0, &qs),
                &slice(dk, &qs),
                &slice(2 * dk, &vs),
                &ov,
                scratch,
                ptr::null()
            ),
            0,
            "{:?}",
            last_error()
        );
        ef_attention_scratch_free(scratch);
        ef_attention_i8_plan_free(plan);
    }
    assert_eq!(out, want);
}

#[test]
fn attention_shapes_and_params_are_checked() {
    assert_eq!(ef_attention_i8_check_shape(2048, 256, 256), 0);
    for (t, dk, dv) in [(2049, 8, 8), (8, 257, 8), (8, 8, 257)] {
        assert_eq!(ef_attention_i8_check_shape(t, dk, dv), -1);
        assert_eq!(errno(), libc::ENOTSUP);
    }
    let x = q(0.1251, -15);
    let good = EfAttentionI8Params {
        q: x,
        q_mul: ptr::null(),
        k: x,
        v: x,
        scores: q(0.1279, -9),
        beta: 1.0,
        probs: out_q(),
        output: q(0.06333, 2),
        kernels: OPT_NEON,
    };
    let mut plan = ptr::null_mut();
    // SAFETY: valid pointers; the values are under test.
    unsafe {
        assert_eq!(ef_attention_i8_plan_new(&good, &mut plan), 0);
        assert_new_fails("scratch over the token limit", libc::ENOTSUP, |o| {
            ef_attention_scratch_new(plan, 1, 4, 2049, 8, 8, o)
        });
        assert_fails_with_null_out("scratch NULL plan", |o| {
            ef_attention_scratch_new(ptr::null(), 1, 4, 8, 8, 8, o)
        });
        assert_eq!(
            ef_attention_scratch_new(plan, 1, 4, 8, 8, 8, ptr::null_mut()),
            -1
        );
        assert_eq!(errno(), libc::EINVAL);
        ef_attention_i8_plan_free(plan);
        assert_new_fails("attention probs quant", libc::ENOTSUP, |o| {
            ef_attention_i8_plan_new(
                &EfAttentionI8Params {
                    probs: q(1.0 / 256.0, 0),
                    ..good
                },
                o,
            )
        });
        assert_fails_with_null_out("attention kernels", |o| {
            ef_attention_i8_plan_new(&EfAttentionI8Params { kernels: 9, ..good }, o)
        });
        assert_fails_with_null_out("attention scale", |o| {
            ef_attention_i8_plan_new(
                &EfAttentionI8Params {
                    k: q(f32::NAN, 0),
                    ..good
                },
                o,
            )
        });
    }
}

#[test]
fn every_new_plan_new_failure_leaves_out_null() {
    let good = EfBinaryI8Params {
        op: EfBinaryOp::Add as u32,
        lhs: q(0.5, 0),
        rhs: q(0.5, 0),
        output: q(1.0, 0),
        activation: ACT_NONE,
        kernels: OPT_NEON,
    };
    // SAFETY: NULL params and bad values are the point; `out` is always writable.
    unsafe {
        assert_fails_with_null_out("binary NULL", |o| ef_binary_i8_plan_new(ptr::null(), o));
        assert_fails_with_null_out("matmul NULL", |o| ef_matmul_i8_plan_new(ptr::null(), o));
        assert_fails_with_null_out("softmax_dot NULL", |o| {
            ef_softmax_dot_i8_plan_new(ptr::null(), o)
        });
        assert_fails_with_null_out("attention NULL", |o| {
            ef_attention_i8_plan_new(ptr::null(), o)
        });
        for (what, p) in [
            ("binary op", EfBinaryI8Params { op: 3, ..good }),
            (
                "binary activation",
                EfBinaryI8Params {
                    activation: 4,
                    ..good
                },
            ),
            ("binary kernels", EfBinaryI8Params { kernels: 4, ..good }),
            (
                "binary scale",
                EfBinaryI8Params {
                    lhs: q(-1.0, 0),
                    ..good
                },
            ),
            (
                "binary zero point",
                EfBinaryI8Params {
                    rhs: q(0.5, 128),
                    ..good
                },
            ),
        ] {
            assert_fails_with_null_out(what, |o| ef_binary_i8_plan_new(&p, o));
        }
    }
    // A valid configuration TFLite's int8 ADD does not take is ENOTSUP.
    let p = EfBinaryI8Params {
        lhs: q(1.0, 0),
        rhs: q(1.0, 0),
        output: q(1.0e-9, 0),
        ..good
    };
    // SAFETY: valid pointers.
    assert_new_fails("binary add ratio", libc::ENOTSUP, |o| unsafe {
        ef_binary_i8_plan_new(&p, o)
    });
}
