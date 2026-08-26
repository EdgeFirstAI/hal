//! `edgefirst-tensor` must carry no process-wide state that affects behaviour.
//!
//! This is the invariant the whole modular design rests on: each C library and
//! each Python extension statically links its own copy of this crate, so any
//! behavioural static is duplicated per artifact and the copies diverge. That
//! is what produced the GL import-cache blocker, and what the mobile-sdk
//! notes record as an iPhone NULL-jump.
//!
//! Benign statics -- probe caches and diagnostics, where N copies cost N
//! probes and nothing else -- are listed here with the reason each is safe.
//! Test-only state (mock structs and locals under `#[cfg(test)] mod ...`)
//! never ships in a linked artifact, so it is out of scope and skipped
//! rather than allowlisted.

use std::path::{Path, PathBuf};

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
        "UNPLANNED_CPU_ACCESS",
        "diagnostic counter; a per-artifact undercount misleads nobody",
    ),
    (
        "WARNED",
        "warn-once dedup set; N copies mean a warning may print once per artifact",
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

/// The name of the static declared on this line, if any.
///
/// Matches `static X`, `pub static X` and `pub(crate) static X` by looking at
/// the first two whitespace tokens rather than requiring the line to *start*
/// with `static ` — an earlier version required that, and `pub static X:
/// Mutex<u64>` slipped through green.
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

/// Process-wide state is declared exactly two ways in Rust: a `static` item,
/// or `thread_local!`. Matching the declaration is both complete and precise.
///
/// Matching *types* or *constructors* instead — `Mutex::new`, `Atomic*::new`,
/// `OnceLock<` — was the earlier approach and it is neither. Those appear in
/// struct-field initialisers and field declarations, which are per-instance
/// and perfectly fine: it flagged `lock: Mutex::new(..)` in `ahardwarebuffer.rs`
/// and `map_state: Mutex::new(..)` in `pbo.rs`, neither of which is shared
/// state. Nothing is lost by dropping them, because a multi-line static's type
/// always follows a line this function already matches.
///
/// A `const` is inlined at each use rather than shared, so it is out of scope.
fn is_state_line(t: &str) -> bool {
    declared_static_name(t).is_some() || t.starts_with("thread_local!")
}

#[test]
fn edgefirst_tensor_declares_no_new_global_state() {
    // Resolved at compile time, so the scan works regardless of the runner's
    // CWD. On a board the source tree is absent entirely -- announce that and
    // return, rather than scanning zero files and reporting a green pass. An
    // earlier version used a relative "src" and passed vacuously on every
    // board, which is the exact failure mode this gate exists to catch.
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    if !src.is_dir() {
        eprintln!(
            "SKIP: {} - source tree not present at {} (deployed test binary)",
            module_path!(),
            src.display()
        );
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
        let lines: Vec<&str> = text.lines().collect();

        // Test-only state (mock structs' Atomic fields, locals inside test
        // functions) never ships in a linked artifact, so `#[cfg(test)] mod
        // ...` blocks are out of scope. Track brace depth from the `mod`
        // line to find where the block ends.
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
            // Exact identifier match, never `contains`: allowlisting `TABLE`
            // by substring also silently permits `INTERN_TABLE`, which is
            // close to the exact type this invariant exists to keep deleted.
            let declared = declared_static_name(t);
            if ALLOWED.iter().any(|(name, _)| {
                declared == Some(*name) || (declared.is_none() && line.contains(name))
            }) {
                continue;
            }
            offenders.push(format!("{}:{}: {}", entry.display(), n + 1, t));
        }
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
