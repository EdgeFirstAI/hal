//! `edgefirst-tensor` must carry no process-wide state that affects behaviour.
//!
//! Scans the crate's sources for `static` and `thread_local!` declarations and
//! fails on any that is not in the allowlist. `#[cfg(test)] mod` blocks are
//! skipped.

use std::path::{Path, PathBuf};

/// Reports a test skip by writing `SKIPPED: {reason}` directly to stderr.
fn report_skip(reason: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "WARNING: SKIPPED: {reason}");
}

const ALLOWED: &[(&str, &str)] = &[
    (
        "SHARED_DRM_FD",
        "N open render-node fds instead of one; no behaviour depends on which",
    ),
    (
        "GET_ID",
        "dlsym cache; N lookups resolve to the same symbol",
    ),
    (
        "SCHEME",
        "compression-support probe cache; same answer per process",
    ),
    ("TABLE", "CUDA dlopen table; dlopen refcounts, same library"),
    (
        "CUDA_IMPORT_SIZE_PADDED",
        "per-copy cache of an adapter property (the driver pads D3D11 allocations); every copy observes the same answer",
    ),
    (
        "CUDA_DEVICE_SELECTED",
        "per-copy cache of cudaSetDevice for the one process D3D11 adapter; \
         every copy computes the same device ordinal, so copies agree",
    ),
    (
        "IMPORTED",
        "per-copy CUDA external-semaphore import of the one process D3D11 \
         fence (cuda.rs `shared_semaphore`); every copy imports the same \
         fence and gets an object with the same meaning, so N copies cost N \
         imports and nothing else. The CUDA stream stays per handle, so no \
         consumer waits behind another's work.",
    ),
    (
        "COMPRESSION_FALLBACKS",
        "diagnostic counter; a per-artifact undercount misleads nobody",
    ),
    (
        "DMA_AVAILABLE",
        "platform availability probe cache; same true/false answer in every process",
    ),
    (
        "IOSURFACE_AVAILABLE",
        "platform availability probe cache; same true/false answer in every process",
    ),
    (
        "AHARDWAREBUFFER_AVAILABLE",
        "platform availability probe cache; same true/false answer in every process",
    ),
    (
        "SHM_AVAILABLE",
        "platform availability probe cache; same true/false answer in every process",
    ),
    (
        "DEVICE",
        "per-copy reference to the one process D3D11 device, its fence and \
         its signal counter (d3d11/device.rs). Not inert, and deliberately \
         so: every copy resolves the same device through the rendezvous \
         mapping `Local\\edgefirst-d3d11-device-<pid>`, so N copies hold N \
         references to one object and agree on its identity -- which is the \
         invariant this guard exists to protect -- rather than each minting \
         a device of its own and diverging.",
    ),
    (
        "EXTERNAL",
        "host-installed device slot of `use_external_device` \
         (d3d11/device.rs); consumed on the first `device()` call and \
         published through the same rendezvous, so N copies of the slot can \
         never yield two devices",
    ),
    (
        "GUARD",
        "opt-in trace-capture install guard (trace.rs). NOT inert, but not a \
         correctness problem either: `tracing-subscriber` is itself statically \
         linked per artifact, so each copy has its OWN subscriber registry -- \
         there is no shared gate arbitrating between them. The real consequence \
         is that start_tracing in one artifact does not capture spans emitted \
         from another's copy, so cross-package traces arrive as separate \
         streams. That is a documented limit of the static-linking model, and \
         it concerns observability rather than behaviour the copies must agree \
         on -- unlike buffer identity, where disagreement renders the wrong \
         texture.",
    ),
    (
        "CPU_MAPPING_GUARD",
        "mapping-exclusion lock for GPU drivers that unmap a range twice \
         (pin.rs). It only delays, never changes a result, so copies cannot \
         disagree about anything. The -capi leaves reach the one copy in \
         libedgefirst_tensor through `ef_tensor_with_cpu_mappings_excluded`; \
         artifacts that each link a static copy exclude only the mappings \
         their own copy makes, the same static-linking limit as GUARD.",
    ),
    (
        "SESSION_USED",
        "opt-in trace-capture single-use marker (trace.rs); same reasoning as GUARD",
    ),
    (
        "ONCE",
        "coverage-only SIGABRT flush handler install guard (covguard.rs); \
         idempotent handler install, test/coverage builds only",
    ),
    (
        "kCFTypeDictionaryKeyCallBacks",
        "extern CoreFoundation symbol; immutable OS-owned constant, not our state",
    ),
    (
        "kCFTypeDictionaryValueCallBacks",
        "extern CoreFoundation symbol; immutable OS-owned constant, not our state",
    ),
    (
        "__EDGEFIRST_COV_INSTALL",
        "coverage constructor, test builds only",
    ),
];

/// Recursively collect every `.rs` file under `dir`.
fn walk_rs_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return out,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(walk_rs_files(&path));
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
    out
}

/// The name of the static declared on this line, if any. Matches `static X`,
/// `pub static X` and `pub(crate) static X`.
fn declared_static_name(t: &str) -> Option<&str> {
    let mut tok = t.split_whitespace();
    let first = tok.next()?;
    let name = if first == "static" {
        tok.next()?
    } else if first == "pub" || first.starts_with("pub(") {
        if tok.next()? != "static" {
            return None;
        }
        tok.next()?
    } else {
        return None;
    };
    Some(name.trim_end_matches(':'))
}

/// Whether the line declares process-wide state: a `static` item or `thread_local!`.
fn is_state_line(t: &str) -> bool {
    declared_static_name(t).is_some() || t.starts_with("thread_local!")
}

/// Scan one source file for process-wide state declarations, skipping
/// `#[cfg(test)] mod ...` blocks.
fn scan_file_for_global_state(path: &Path, text: &str, offenders: &mut Vec<String>) {
    let lines: Vec<&str> = text.lines().collect();

    // Tracks brace depth from a `#[cfg(test)] mod` line to find where the block ends.
    let mut pending_test_cfg = false;
    let mut skip_depth: i32 = 0;

    for (n, line) in lines.iter().enumerate() {
        let t = line.trim_start();

        if skip_depth > 0 {
            skip_depth += line.matches('{').count() as i32;
            skip_depth -= line.matches('}').count() as i32;
            continue;
        }

        if t == "#[cfg(test)]" {
            pending_test_cfg = true;
            continue;
        }
        if pending_test_cfg {
            pending_test_cfg = false;
            if t.starts_with("mod ") {
                let depth = line.matches('{').count() as i32 - line.matches('}').count() as i32;
                if depth > 0 {
                    skip_depth = depth;
                    continue;
                }
            }
        }

        if !is_state_line(t) {
            continue;
        }
        // Exact identifier match against the allowlist.
        let declared = declared_static_name(t);
        if ALLOWED
            .iter()
            .any(|(name, _)| declared == Some(*name) || (declared.is_none() && line.contains(name)))
        {
            continue;
        }
        offenders.push(format!("{}:{}: {}", path.display(), n + 1, t));
    }
}

#[test]
fn edgefirst_tensor_declares_no_new_global_state() {
    // Resolved at compile time. When the source tree is absent the test reports a skip.
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    if !src.is_dir() {
        report_skip(&format!(
            "{} - source tree not present at {} (deployed test binary)",
            module_path!(),
            src.display()
        ));
        return;
    }

    let files = walk_rs_files(&src);
    assert!(
        !files.is_empty(),
        "scanned zero .rs files under {} -- the gate would pass without \
         inspecting anything",
        src.display()
    );

    let mut offenders = Vec::new();
    for entry in files {
        let text = std::fs::read_to_string(&entry).unwrap();
        scan_file_for_global_state(&entry, &text, &mut offenders);
    }
    assert!(
        offenders.is_empty(),
        "new process-wide state in edgefirst-tensor:\n  {}\n\n\
         Each C library and Python extension links its own copy of this crate, so \
         a behavioural static is duplicated per artifact and the copies diverge. \
         If this one is genuinely benign, add it to ALLOWED with the reason.",
        offenders.join("\n  ")
    );
}
