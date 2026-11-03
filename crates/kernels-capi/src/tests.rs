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
        "EfKernel",
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

fn load_yolo11() -> Fixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/kernels/softmax_i8_yolo11_attn.safetensors"
    );
    let bytes = std::fs::read(path).expect("fixture present");
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

/// Pre-sets `*out` to a dangling non-NULL value, runs `new`, and checks failure leaves NULL.
fn assert_fails_with_null_out<P>(what: &str, new: impl FnOnce(*mut *mut P) -> i32) {
    let mut out: *mut P = ptr::NonNull::dangling().as_ptr();
    assert_eq!(new(&mut out), -1, "{what}");
    assert_eq!(errno(), libc::EINVAL, "{what}");
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
