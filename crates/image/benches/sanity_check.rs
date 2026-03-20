// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Adaptive sanity check for letterbox format combinations.
//!
//! Each test runs in a forked child process with a 30-second alarm timeout
//! to detect GPU hangs without blocking the parent. Results are reported
//! via exit code and a small result file.
//!
//! - Run 1 iteration first
//! - If >100ms, run 10 more iterations
//! - If median >1000ms after 10, flag as TOO_SLOW
//! - If child killed by SIGALRM, flag as GPU_HANG
//!
//! Usage:
//! ```bash
//! EDGEFIRST_FORCE_BACKEND=opengl ./sanity_check
//! EDGEFIRST_FORCE_BACKEND=cpu ./sanity_check
//! EDGEFIRST_FORCE_BACKEND=g2d ./sanity_check
//! ```

mod common;

use common::{calculate_letterbox, format_name, get_test_data, BenchConfig};
use edgefirst_image::{
    ComputeBackend, Crop, Flip, ImageProcessor, ImageProcessorConfig, ImageProcessorTrait, Rect,
    Rotation,
};
use edgefirst_tensor::PixelFormat;
use edgefirst_tensor::{TensorMapTrait, TensorTrait};
use std::io::{Read as _, Write as _};
use std::time::{Duration, Instant};

const TIMEOUT_SECS: u32 = 30;

fn median(times: &mut [Duration]) -> Duration {
    times.sort();
    let mid = times.len() / 2;
    if times.len().is_multiple_of(2) {
        (times[mid - 1] + times[mid]) / 2
    } else {
        times[mid]
    }
}

/// Run a single test in the current process and write results to the given fd.
fn run_single_test(backend: ComputeBackend, config: &BenchConfig, result_fd: i32) {
    // Set alarm for GPU hang detection
    unsafe { libc::alarm(TIMEOUT_SECS) };

    // `needless_update` fires on macOS where `ImageProcessorConfig` has only
    // `backend`, but on Linux the struct also has the cfg-gated `egl_display`.
    #[allow(clippy::needless_update)]
    let ipc_config = ImageProcessorConfig {
        backend,
        ..Default::default()
    };
    let mut proc = match ImageProcessor::with_config(ipc_config) {
        Ok(p) => p,
        Err(e) => {
            write_result(result_fd, &format!("FAILED:init:{e}:0:0:0"));
            return;
        }
    };

    let src = match proc.create_image(config.in_w, config.in_h, config.in_fmt, None) {
        Ok(s) => s,
        Err(e) => {
            write_result(result_fd, &format!("FAILED:src_alloc:{e}:0:0:0"));
            return;
        }
    };
    let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
    src.as_u8().unwrap().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);

    let mut dst = match proc.create_image(config.out_w, config.out_h, config.out_fmt, None) {
        Ok(d) => d,
        Err(e) => {
            write_result(result_fd, &format!("FAILED:dst_alloc:{e}:0:0:0"));
            return;
        }
    };

    let (left, top, new_w, new_h) =
        calculate_letterbox(config.in_w, config.in_h, config.out_w, config.out_h);
    let crop = Crop::new()
        .with_dst_rect(Some(Rect::new(left, top, new_w, new_h)))
        .with_dst_color(Some([114, 114, 114, 255]));

    // Phase 1: single iteration
    let start = Instant::now();
    if let Err(e) = proc.convert(&src, &mut dst, Rotation::None, Flip::None, crop) {
        write_result(result_fd, &format!("FAILED:convert:{e}:0:0:0"));
        return;
    }
    let single_ms = start.elapsed().as_secs_f64() * 1000.0;

    if single_ms <= 100.0 {
        write_result(result_fd, &format!("OK:{single_ms:.1}:0:1"));
        return;
    }

    if single_ms > 1000.0 {
        write_result(result_fd, &format!("TOO_SLOW:{single_ms:.1}:0:1"));
        return;
    }

    // Phase 2: 10 more iterations
    let mut times = Vec::with_capacity(10);
    for i in 0..10 {
        let start = Instant::now();
        if let Err(e) = proc.convert(&src, &mut dst, Rotation::None, Flip::None, crop) {
            write_result(
                result_fd,
                &format!("FAILED:iter_{i}:{e}:{single_ms:.1}:0:{}", i + 1),
            );
            return;
        }
        times.push(start.elapsed());
    }

    let med = median(&mut times);
    let med_ms = med.as_secs_f64() * 1000.0;

    let status = if med_ms > 1000.0 {
        "TOO_SLOW"
    } else if med_ms > 100.0 {
        "SLOW"
    } else {
        "OK"
    };
    write_result(
        result_fd,
        &format!("{status}:{single_ms:.1}:{med_ms:.1}:11"),
    );
}

fn write_result(fd: i32, msg: &str) {
    use std::os::unix::io::FromRawFd;
    let mut f = unsafe { std::fs::File::from_raw_fd(fd) };
    let _ = f.write_all(msg.as_bytes());
    // Don't drop f — it'll close the fd, which is what we want
}

/// Fork a child, run the test, collect results with timeout.
fn run_test_forked(
    backend: ComputeBackend,
    config: &BenchConfig,
    backend_name: &str,
) -> (String, String, String, String, usize, String) {
    let name = format!(
        "letterbox/{}x{}/{}->{}/{}x{}",
        config.in_w,
        config.in_h,
        format_name(config.in_fmt),
        format_name(config.out_fmt),
        config.out_w,
        config.out_h,
    );

    // Create a pipe for result communication
    let mut pipe_fds = [0i32; 2];
    if unsafe { libc::pipe(pipe_fds.as_mut_ptr()) } != 0 {
        return (
            name,
            backend_name.to_string(),
            "—".into(),
            "—".into(),
            0,
            "FAILED: pipe".into(),
        );
    }
    let read_fd = pipe_fds[0];
    let write_fd = pipe_fds[1];

    let pid = unsafe { libc::fork() };
    match pid {
        -1 => {
            unsafe {
                libc::close(read_fd);
                libc::close(write_fd);
            }
            (
                name,
                backend_name.to_string(),
                "—".into(),
                "—".into(),
                0,
                "FAILED: fork".into(),
            )
        }
        0 => {
            // Child process
            unsafe { libc::close(read_fd) };
            run_single_test(backend, config, write_fd);
            unsafe { libc::_exit(0) };
        }
        child_pid => {
            // Parent process
            unsafe { libc::close(write_fd) };

            // Wait for child with timeout
            let wait_start = Instant::now();
            let timeout = Duration::from_secs((TIMEOUT_SECS + 5) as u64);
            let mut status: i32 = 0;

            loop {
                let ret = unsafe { libc::waitpid(child_pid, &mut status, libc::WNOHANG) };
                if ret == child_pid {
                    break;
                }
                if ret == -1 {
                    break;
                }
                if wait_start.elapsed() > timeout {
                    // Kill the child
                    unsafe { libc::kill(child_pid, libc::SIGKILL) };
                    unsafe { libc::waitpid(child_pid, &mut status, 0) };
                    unsafe { libc::close(read_fd) };
                    return (
                        name,
                        backend_name.to_string(),
                        "—".into(),
                        "—".into(),
                        0,
                        "GPU_HANG (timeout)".into(),
                    );
                }
                std::thread::sleep(Duration::from_millis(100));
            }

            // Read result from pipe
            let mut result_buf = String::new();
            {
                use std::os::unix::io::FromRawFd;
                let mut f = unsafe { std::fs::File::from_raw_fd(read_fd) };
                let _ = f.read_to_string(&mut result_buf);
                // f drops here, closing read_fd
            }

            // Check if child was killed by signal
            if libc::WIFSIGNALED(status) {
                let sig = libc::WTERMSIG(status);
                let sig_name = match sig {
                    14 => "SIGALRM (GPU hang)",
                    9 => "SIGKILL (timeout)",
                    11 => "SIGSEGV (crash)",
                    _ => "unknown signal",
                };
                return (
                    name,
                    backend_name.to_string(),
                    "—".into(),
                    "—".into(),
                    0,
                    format!("GPU_HANG: {sig_name}"),
                );
            }

            // Parse result: STATUS:single_ms:med_ms:iters  or  FAILED:reason:detail:...
            parse_result(&name, backend_name, &result_buf)
        }
    }
}

fn parse_result(
    name: &str,
    backend_name: &str,
    result: &str,
) -> (String, String, String, String, usize, String) {
    let parts: Vec<&str> = result.splitn(4, ':').collect();
    if parts.is_empty() {
        return (
            name.to_string(),
            backend_name.to_string(),
            "—".into(),
            "—".into(),
            0,
            "FAILED: no result".into(),
        );
    }

    let status = parts[0];
    if status == "FAILED" {
        let reason = parts.get(1).unwrap_or(&"unknown");
        let detail = parts.get(2).unwrap_or(&"");
        return (
            name.to_string(),
            backend_name.to_string(),
            "—".into(),
            "—".into(),
            0,
            format!("FAILED: {reason}: {detail}"),
        );
    }

    // OK/SLOW/TOO_SLOW:single_ms:med_ms:iters
    let single = parts.get(1).unwrap_or(&"—").to_string();
    let med = parts.get(2).unwrap_or(&"—").to_string();
    let iters_str = parts.get(3).unwrap_or(&"0");
    let iters: usize = iters_str.parse().unwrap_or(0);
    let med_display = if med == "0" { "—".to_string() } else { med };

    (
        name.to_string(),
        backend_name.to_string(),
        single,
        med_display,
        iters,
        status.to_string(),
    )
}

fn main() {
    let backend_env = std::env::var("EDGEFIRST_FORCE_BACKEND").unwrap_or_default();

    let backends: Vec<(ComputeBackend, &str)> = if !backend_env.is_empty() {
        let b = match backend_env.as_str() {
            "cpu" => ComputeBackend::Cpu,
            "opengl" => ComputeBackend::OpenGl,
            "g2d" => ComputeBackend::G2d,
            _ => {
                eprintln!("Unknown backend: {backend_env}");
                std::process::exit(1);
            }
        };
        vec![(b, backend_env.as_str())]
    } else {
        // Test all backends sequentially
        vec![
            (ComputeBackend::Cpu, "cpu"),
            (ComputeBackend::OpenGl, "opengl"),
            (ComputeBackend::G2d, "g2d"),
        ]
    };

    // Test safe combos first, known GPU-hang combos last (PixelFormat::Nv12→planar on Vivante).
    // Each (input, output) pair — ordered so GPU-hang candidates are last.
    let test_combos: Vec<(PixelFormat, PixelFormat)> = vec![
        // Yuyv sources — all safe on Vivante
        (PixelFormat::Yuyv, PixelFormat::Rgba),
        (PixelFormat::Yuyv, PixelFormat::Rgb),
        (PixelFormat::Yuyv, PixelFormat::PlanarRgb),
        // Nv12 sources — packed/Rgba are safe, planar may hang
        (PixelFormat::Nv12, PixelFormat::Rgba),
        (PixelFormat::Nv12, PixelFormat::Rgb),
        // Nv12 → planar: KNOWN GPU HANG RISK on Vivante GC7000UL
        (PixelFormat::Nv12, PixelFormat::PlanarRgb),
    ];
    let resolutions = [(1920, 1080, "1080p"), (1280, 720, "720p")];

    println!(
        "=== Letterbox Sanity Check (forked, {}s timeout per test) ===",
        TIMEOUT_SECS
    );
    println!(
        "Date: {}",
        std::process::Command::new("date")
            .arg("-Iseconds")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|_| "unknown".into())
    );
    println!();

    println!(
        "{:<55} {:<8} {:>10} {:>12} {:>6} {:<30}",
        "Test", "Backend", "1st (ms)", "Med/10 (ms)", "Iters", "Status"
    );
    println!("{}", "-".repeat(125));

    let mut all_results = Vec::new();

    for (backend, backend_name) in &backends {
        for (res_w, res_h, _res_name) in &resolutions {
            for (in_fmt, out_fmt) in &test_combos {
                let config = BenchConfig::new(*res_w, *res_h, 640, 640, *in_fmt, *out_fmt);
                let (name, bk, single, med, iters, status) =
                    run_test_forked(*backend, &config, backend_name);

                println!(
                    "{:<55} {:<8} {:>10} {:>12} {:>6} {:<30}",
                    name, bk, single, med, iters, status,
                );

                all_results.push((name, bk, single, med, iters, status));
            }
        }
    }

    println!();
    println!("=== Summary ===");
    let ok = all_results.iter().filter(|r| r.5 == "OK").count();
    let slow = all_results.iter().filter(|r| r.5 == "SLOW").count();
    let too_slow = all_results.iter().filter(|r| r.5 == "TOO_SLOW").count();
    let hang = all_results
        .iter()
        .filter(|r| r.5.starts_with("GPU_HANG"))
        .count();
    let failed = all_results
        .iter()
        .filter(|r| r.5.starts_with("FAILED"))
        .count();
    let total = all_results.len();
    println!("Total: {total}  OK: {ok}  SLOW: {slow}  TOO_SLOW: {too_slow}  GPU_HANG: {hang}  FAILED: {failed}");

    if too_slow > 0 || hang > 0 || failed > 0 {
        println!();
        println!("=== Problem Combinations ===");
        for r in &all_results {
            if r.5 != "OK" && r.5 != "SLOW" {
                println!("  {} [{}] — {}", r.0, r.1, r.5);
            }
        }
    }
}
