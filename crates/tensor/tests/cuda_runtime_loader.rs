// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The CUDA runtime loader against a real libcudart.
//!
//! Without a runtime this skips; under `HAL_TEST_REQUIRE_CUDA=1` that skip is
//! a failure, so a lane meant to have CUDA cannot report the loader as tested
//! when it never loaded. The no-runtime half (every primitive degrades to
//! `None`/`false`) is a unit test in `src/cuda.rs`.

/// Only `decide` is used: `cuda_available_or_skip` words its reason for the
/// Windows DLL probe.
#[path = "support/cuda_require.rs"]
#[allow(dead_code)]
mod cuda_require;

fn cuda_required() -> bool {
    std::env::var("HAL_TEST_REQUIRE_CUDA").as_deref() == Ok("1")
}

fn cuda_or_skip(what: &str) -> bool {
    use std::io::Write;
    let why = "no CUDA runtime (libcudart / cudart64_*.dll not found by the loader)";
    let run = cuda_require::decide(
        edgefirst_tensor::is_cuda_available(),
        cuda_required(),
        what,
        why,
    );
    if !run {
        let _ = writeln!(&mut std::io::stderr(), "SKIPPED: {what} - {why}");
    }
    run
}

#[test]
fn a_loaded_cuda_runtime_names_its_library_and_creates_a_stream() {
    const NAME: &str = "a_loaded_cuda_runtime_names_its_library_and_creates_a_stream";
    if !cuda_or_skip(NAME) {
        return;
    }
    let path = edgefirst_tensor::cuda_runtime_path()
        .expect("runtime_path is set whenever the runtime loaded");
    assert!(
        !path.as_os_str().is_empty(),
        "runtime_path must name the library the loader opened"
    );
    // Straight to stderr so libtest's capture keeps it for a passing test.
    let _ = std::io::Write::write_fmt(
        &mut std::io::stderr(),
        format_args!("[cuda] runtime_path = {}\n", path.display()),
    );

    // A library without a usable device fails stream creation, so demand a
    // stream only where CUDA is required (the lane has a GPU).
    match edgefirst_tensor::stream_create() {
        Some(stream) => {
            assert!(!stream.is_null(), "a created stream is a real handle");
            // SAFETY: `stream` is live and destroyed exactly once.
            unsafe {
                assert!(edgefirst_tensor::stream_synchronize(stream));
                edgefirst_tensor::stream_destroy(stream);
            }
        }
        None => assert!(
            !cuda_required(),
            "HAL_TEST_REQUIRE_CUDA=1 but the loaded runtime could not create a stream"
        ),
    }
}
