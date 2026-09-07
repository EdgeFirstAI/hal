// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//! MSVC toolchain discovery for the C tests on Windows.
//!
//! There is no `cc` on a Windows host, and `cl.exe` is only on PATH inside
//! a Visual Studio developer shell. This module finds a working `cl.exe`
//! once per test process: the current environment when it already is a
//! developer shell, otherwise the newest Visual Studio with the C++ toolset
//! (through vswhere), whose `VsDevCmd.bat` supplies PATH, INCLUDE and LIB.
//! That is the bootstrap `crates/gpu-probe/windows/build.ps1` uses.
//!
//! Discovery ends with a trivial program compiled and linked, so a caller
//! that gets a [`Toolchain`] has one that works. Everything that can fail
//! carries the command line and the compiler output into the reason string,
//! because a skip whose reason is a bare "cannot build" cannot be acted on.
//!
//! Discovery also sets the process error mode to
//! `SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX`. A C program that fails
//! an `assert()` aborts through `_CALL_REPORTFAULT`, and Windows Error
//! Reporting would put a dialog in front of it; a child inherits the error
//! mode, so setting it here turns that dialog into an exit code the harness
//! can report instead of a test that hangs in `output()`.
//!
//! Test support only, `#[path]`-included by each modular C-API leaf's test
//! module; nothing here reaches a shipped library.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

/// System libraries a leaf's Rust staticlib needs on top of its own archive.
///
/// Captured with `cargo rustc --manifest-path crates/<leaf>-capi/Cargo.toml
/// --lib -- --print native-static-libs` on rustc 1.98.0, which prints the
/// same list for both leaves (the image leaf adds `edgefirst_tensor.dll.lib`,
/// which its caller passes, and repeats `kernel32.lib`). `/defaultlib:msvcrt`
/// from the same note is why the harness compiles with `/MD`.
///
/// Direct3D and DXGI are absent from that note because the `windows` crate
/// binds them as raw-dylib imports, which travel inside the staticlib
/// itself; neither leaf names `d3d11.lib` or `dxgi.lib` on its link line for
/// that reason.
pub const SYSTEM_LIBS: &[&str] = &[
    "legacy_stdio_definitions.lib",
    "kernel32.lib",
    "ntdll.lib",
    "userenv.lib",
    "ws2_32.lib",
    "dbghelp.lib",
];

/// Compiler flags a caller's environment must not be able to inject.
///
/// `cl.exe` prepends `CL` and appends `_CL_` to every command line it is
/// given, so a developer with `CL=/w` set would silently defeat the
/// `/W4 /WX` these checks depend on.
const INJECTED_FLAG_VARS: [&str; 2] = ["CL", "_CL_"];

// SAFETY: kernel32 exports these with the documented signatures; the harness
// is Windows-only test-support code and the `windows` crate stays inside the
// tensor crate.
#[link(name = "kernel32", kind = "raw-dylib")]
unsafe extern "system" {
    fn SetErrorMode(mode: u32) -> u32;
    fn OpenProcess(access: u32, inherit: i32, pid: u32) -> *mut std::ffi::c_void;
    fn CloseHandle(handle: *mut std::ffi::c_void) -> i32;
    fn GetLastError() -> u32;
}

const SEM_FAILCRITICALERRORS: u32 = 0x0001;
const SEM_NOGPFAULTERRORBOX: u32 = 0x0002;
const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
const ERROR_INVALID_PARAMETER: u32 = 87;

/// A `cl.exe` proven to build a trivial program, plus the environment it
/// needs to find headers and libraries and a scratch directory to write to.
pub struct Toolchain {
    cl: PathBuf,
    env: Vec<(String, String)>,
    scratch: PathBuf,
}

/// Report a skipped check on stderr.
///
/// Straight to the handle, not `eprintln!`: libtest captures a passing
/// test's `eprintln!` and a skip nobody sees is indistinguishable from a
/// pass.
pub fn skip(reason: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "SKIP: {reason}");
}

/// The toolchain for this process, or the reason there is none.
///
/// Located once: discovery runs `VsDevCmd.bat` and a probe compile, and
/// repeating either per test both wastes seconds and races, since several
/// libtest threads would drive one compiler over one set of scratch files.
pub fn toolchain() -> Result<&'static Toolchain, &'static str> {
    static TOOLCHAIN: OnceLock<Result<Toolchain, String>> = OnceLock::new();
    TOOLCHAIN
        .get_or_init(locate)
        .as_ref()
        .map_err(|reason| reason.as_str())
}

/// The toolchain, or `None` after saying on stderr why `what` is skipped.
pub fn require(what: &str) -> Option<&'static Toolchain> {
    match toolchain() {
        Ok(toolchain) => Some(toolchain),
        Err(reason) => {
            skip(&format!("{what} not compiled: no usable MSVC -- {reason}"));
            None
        }
    }
}

impl Toolchain {
    /// A private directory for this process's compiler output.
    ///
    /// Per-process, not a fixed name under `%TEMP%`: two leaves' test
    /// binaries, or two `cargo test` runs, would otherwise compile over
    /// each other's object and executable files. It outlives the run so a
    /// failure can be reproduced by hand; [`sweep_scratch`] removes the
    /// directories of processes that have exited.
    pub fn scratch(&self) -> &Path {
        &self.scratch
    }

    /// `cl.exe` with the developer environment applied.
    fn cl(&self) -> Command {
        let mut cmd = self.tool(&self.cl);
        cmd.arg("/nologo");
        cmd
    }

    /// `tool` with this toolchain's environment applied.
    ///
    /// The vswhere branch captured a whole environment from `VsDevCmd.bat`,
    /// which already carries everything a compiler needs, so the caller's is
    /// cleared first: nothing outside that capture should reach the
    /// compiler. The developer-shell branch captured nothing and keeps
    /// inheriting, because its environment *is* the developer environment.
    /// Either way the two variables `cl.exe` splices into its own command
    /// line are removed.
    fn tool(&self, tool: &Path) -> Command {
        let mut cmd = Command::new(tool);
        if !self.env.is_empty() {
            cmd.env_clear();
            cmd.envs(
                self.env
                    .iter()
                    .filter(|(k, _)| !INJECTED_FLAG_VARS.iter().any(|v| k.eq_ignore_ascii_case(v)))
                    .map(|(k, v)| (k, v)),
            );
        }
        for var in INJECTED_FLAG_VARS {
            cmd.env_remove(var);
        }
        cmd
    }

    /// Parse `src` without generating code.
    ///
    /// `/W4 /WX` is the `-Wall -Werror` of the POSIX branch; `flags` carries
    /// per-check options such as `/std:c11`.
    pub fn syntax_check(&self, includes: &[&str], flags: &[&str], src: &str) -> Output {
        let mut cmd = self.cl();
        cmd.args(["/Zs", "/W4", "/WX"]).args(flags);
        for dir in includes {
            cmd.arg("/I").arg(dir);
        }
        cmd.arg(src);
        run(cmd)
    }

    /// Compile `src` and link it with `libs` and [`SYSTEM_LIBS`] into `exe`.
    ///
    /// `/MD` matches the dynamic CRT rustc's staticlib defaults to
    /// (`/defaultlib:msvcrt` in its link note); a static-CRT object next to
    /// it would give the linker two C runtimes.
    pub fn build(&self, includes: &[&str], src: &str, exe: &Path, libs: &[&OsStr]) -> Output {
        let mut cmd = self.cl();
        cmd.args(["/W4", "/WX", "/MD"]);
        for dir in includes {
            cmd.arg("/I").arg(dir);
        }
        cmd.arg(src)
            .arg(format!("/Fe:{}", exe.display()))
            .arg(format!("/Fo:{}", exe.with_extension("obj").display()))
            .arg("/link")
            .args(libs)
            .args(SYSTEM_LIBS);
        run(cmd)
    }

    /// Names `binary` exports, from `dumpbin /exports`.
    ///
    /// The Windows counterpart of `nm -D --defined-only`. `dumpbin.exe` sits
    /// beside `cl.exe` in the toolset's `bin` directory.
    pub fn exports(&self, binary: &Path) -> Result<Vec<String>, String> {
        let dumpbin = self.cl.with_file_name("dumpbin.exe");
        if !dumpbin.is_file() {
            return Err(format!("{} not found", dumpbin.display()));
        }
        let mut cmd = self.tool(&dumpbin);
        cmd.args(["/nologo", "/exports"]).arg(binary);
        let out = run(cmd);
        if !out.status.success() {
            return Err(String::from_utf8_lossy(&out.stderr).into_owned());
        }
        // Each export is one `ordinal hint RVA name` row under a header of
        // those four words; the decimal ordinal and the hexadecimal RVA
        // separate those rows from every other line of the dump.
        let text = String::from_utf8_lossy(&out.stdout);
        let mut names: Vec<String> = text
            .lines()
            .filter_map(|line| {
                let mut fields = line.split_whitespace();
                let ordinal = fields.next()?;
                let hint = fields.next()?;
                let rva = fields.next()?;
                let name = fields.next()?;
                let hex = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_hexdigit());
                (ordinal.parse::<u32>().is_ok() && hex(hint) && hex(rva)).then(|| name.to_string())
            })
            .collect();
        names.sort();
        names.dedup();
        if names.is_empty() {
            return Err(format!("{} exports nothing", binary.display()));
        }
        Ok(names)
    }
}

/// Run `cmd` and fold what it printed into one `stderr`.
///
/// `cl.exe` and `link.exe` write their diagnostics to stdout, so a caller
/// that reports only `stderr` would show nothing at all for a failed
/// compile. The command line leads the text so a report says what ran.
///
/// A tool that cannot be spawned comes back as a failed `Output` rather
/// than a panic: discovery runs inside a `OnceLock`, where a panic would
/// poison the cell for every later caller instead of reaching the skip
/// path that exists to report exactly this.
fn run(mut cmd: Command) -> Output {
    let line = format!("{cmd:?}");
    match cmd.output() {
        Ok(out) => {
            let mut stderr = format!("$ {line}\n").into_bytes();
            stderr.extend_from_slice(&out.stdout);
            stderr.extend_from_slice(&out.stderr);
            Output {
                status: out.status,
                stdout: out.stdout,
                stderr,
            }
        }
        Err(e) => failed_output(&format!("$ {line}\ncannot be run: {e}")),
    }
}

/// An `Output` that failed, carrying `reason` as its `stderr`.
fn failed_output(reason: &str) -> Output {
    use std::os::windows::process::ExitStatusExt;
    Output {
        status: std::process::ExitStatus::from_raw(1),
        stdout: Vec::new(),
        stderr: reason.as_bytes().to_vec(),
    }
}

fn locate() -> Result<Toolchain, String> {
    quiet_crash_dialogs();
    sweep_scratch();
    let scratch = scratch_dir()?;
    let mut reasons = Vec::new();
    // A developer shell already carries everything cl.exe needs. Its cl.exe
    // is still probed, and a failure falls through to vswhere rather than
    // ending discovery: a half-configured shell must not hide a working
    // install.
    if std::env::var_os("INCLUDE").is_some() && std::env::var_os("LIB").is_some() {
        if let Some(cl) = find_on_path(std::env::var_os("PATH").as_deref()) {
            let candidate = Toolchain {
                cl,
                env: Vec::new(),
                scratch: scratch.clone(),
            };
            match probe(&candidate) {
                Ok(()) => return Ok(candidate),
                Err(reason) => reasons.push(format!("this developer shell: {reason}")),
            }
        }
    }
    let vs = visual_studio_install().map_err(|e| joined(&reasons, &e))?;
    let env = developer_environment(&vs).map_err(|e| joined(&reasons, &e))?;
    let path = env
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("PATH"))
        .map(|(_, v)| OsString::from(v));
    let cl = find_on_path(path.as_deref()).ok_or_else(|| {
        joined(
            &reasons,
            &format!("VsDevCmd.bat in {} put no cl.exe on PATH", vs.display()),
        )
    })?;
    let candidate = Toolchain { cl, env, scratch };
    probe(&candidate).map_err(|e| joined(&reasons, &e))?;
    Ok(candidate)
}

/// One reason string out of the candidates that were tried.
fn joined(earlier: &[String], last: &str) -> String {
    if earlier.is_empty() {
        return last.to_string();
    }
    format!("{}; {last}", earlier.join("; "))
}

/// Turns a crashing child's error dialogs into an exit code.
///
/// A failed `assert()` in one of these C programs calls `abort()`, which
/// hands the process to Windows Error Reporting; on a desktop that is a
/// modal dialog, and the harness would sit in `output()` until someone
/// dismissed it. Children inherit the error mode, so one call before the
/// first spawn covers every program.
fn quiet_crash_dialogs() {
    // SAFETY: documented call with no preconditions; the previous mode is
    // discarded because nothing in a test process depends on it.
    unsafe { SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX) };
}

/// The prefix every scratch directory's name starts with.
const SCRATCH_PREFIX: &str = "edgefirst-capi-c-";

/// A per-process directory under `%TEMP%` for probe and test binaries.
fn scratch_dir() -> Result<PathBuf, String> {
    let dir = std::env::temp_dir().join(format!("{SCRATCH_PREFIX}{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    Ok(dir)
}

/// Removes the scratch directories of test processes that have exited.
///
/// The current process keeps its own until the next run sweeps it, so a
/// failing program can be re-run by hand from the path the diagnostic
/// names. libtest offers no shutdown hook, and a `static` is never dropped,
/// so a sweep at discovery is what keeps `%TEMP%` from growing.
fn sweep_scratch() {
    let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
        return;
    };
    let me = std::process::id();
    for entry in entries.filter_map(|e| e.ok()) {
        let name = entry.file_name().to_string_lossy().to_string();
        let Some(pid) = name
            .strip_prefix(SCRATCH_PREFIX)
            .and_then(|p| p.parse::<u32>().ok())
        else {
            continue;
        };
        if pid == me || is_process_alive(pid) {
            continue;
        }
        // A directory another process is still writing is left alone by the
        // liveness check above; a failure here (a file held open by a
        // debugger, say) is not worth reporting.
        let _ = std::fs::remove_dir_all(entry.path());
    }
}

/// Is a process with this id still running?
///
/// Only a handle the kernel refuses because the id names nothing counts as
/// gone: a live process this one may not query is still live, and sweeping
/// its directory would delete files it is compiling into.
fn is_process_alive(pid: u32) -> bool {
    // SAFETY: documented call; a returned handle is closed below.
    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if !handle.is_null() {
        // SAFETY: `handle` came from OpenProcess and is not used again.
        unsafe { CloseHandle(handle) };
        return true;
    }
    // SAFETY: reads the calling thread's last error, set by the call above.
    unsafe { GetLastError() != ERROR_INVALID_PARAMETER }
}

/// Compile and link a trivial program with this toolchain.
///
/// Separates "MSVC is broken on this host" from "our library is broken",
/// the same split the POSIX harness makes with `cc`. The error carries the
/// command line and everything the compiler printed.
fn probe(toolchain: &Toolchain) -> Result<(), String> {
    let scratch = &toolchain.scratch;
    let src = scratch.join("probe.c");
    std::fs::write(&src, b"int main(void){return 0;}\n")
        .map_err(|e| format!("cannot write {}: {e}", src.display()))?;
    let mut cmd = toolchain.cl();
    cmd.arg(&src)
        .arg(format!("/Fe:{}", scratch.join("probe.exe").display()))
        .arg(format!("/Fo:{}", scratch.join("probe.obj").display()));
    let out = run(cmd);
    if out.status.success() {
        return Ok(());
    }
    Err(format!(
        "{} cannot build a trivial program ({}):\n{}",
        toolchain.cl.display(),
        out.status,
        String::from_utf8_lossy(&out.stderr).trim_end()
    ))
}

fn find_on_path(path: Option<&OsStr>) -> Option<PathBuf> {
    std::env::split_paths(path?)
        .map(|dir| dir.join("cl.exe"))
        .find(|cl| cl.is_file())
}

/// The newest Visual Studio or Build Tools install with the C++ toolset.
fn visual_studio_install() -> Result<PathBuf, String> {
    let vswhere = find_vswhere()?;
    let out = Command::new(&vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ])
        .output()
        .map_err(|e| format!("{}: {e}", vswhere.display()))?;
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if !out.status.success() || path.is_empty() {
        return Err(format!(
            "{} found no Visual Studio with the C++ toolset ({}):\n{}",
            vswhere.display(),
            out.status,
            String::from_utf8_lossy(&out.stderr).trim_end()
        ));
    }
    Ok(PathBuf::from(path))
}

/// vswhere's fixed install location under the 32-bit program files.
///
/// `ProgramFiles(x86)` is the documented way to name that directory, but a
/// parenthesis in a variable name does not survive every shell, so the
/// literal path is tried as well rather than reporting no toolchain.
fn find_vswhere() -> Result<PathBuf, String> {
    let mut roots: Vec<PathBuf> = Vec::new();
    for var in ["ProgramFiles(x86)", "ProgramFiles"] {
        if let Some(root) = std::env::var_os(var) {
            roots.push(PathBuf::from(root));
        }
    }
    let system_drive = std::env::var("SystemDrive").unwrap_or_else(|_| "C:".to_string());
    roots.push(PathBuf::from(format!(
        "{system_drive}\\Program Files (x86)"
    )));
    let mut tried = Vec::new();
    for root in roots {
        let vswhere = root.join(r"Microsoft Visual Studio\Installer\vswhere.exe");
        if vswhere.is_file() {
            return Ok(vswhere);
        }
        tried.push(vswhere.display().to_string());
    }
    Err(format!(
        "vswhere.exe not found; looked in {}",
        tried.join(", ")
    ))
}

/// Run `VsDevCmd.bat` and capture the environment it leaves behind.
fn developer_environment(vs: &Path) -> Result<Vec<(String, String)>, String> {
    use std::os::windows::process::CommandExt;
    let bat = vs.join(r"Common7\Tools\VsDevCmd.bat");
    if !bat.is_file() {
        return Err(format!("{} not found", bat.display()));
    }
    let arch = if cfg!(target_arch = "aarch64") {
        "arm64"
    } else {
        "amd64"
    };
    // Passed verbatim: cmd's own quoting rules, not the C runtime's. `/u`
    // makes cmd write UTF-16, so an install or profile path outside the OEM
    // code page survives; `/s` strips the outer quotes and runs the quoted
    // batch path with its arguments; `set` then prints the environment the
    // batch left behind.
    let line = format!(
        "/d /u /s /c \"\"{}\" -arch={arch} -host_arch={arch} -no_logo && set\"",
        bat.display()
    );
    let out = Command::new("cmd.exe")
        .raw_arg(&line)
        .output()
        .map_err(|e| format!("cmd.exe {line}: {e}"))?;
    let stdout = decode_console(&out.stdout);
    if !out.status.success() {
        return Err(format!(
            "cmd.exe {line} exited {}:\n{}{}",
            out.status,
            banner(&stdout),
            decode_console(&out.stderr).trim_end()
        ));
    }
    // `set` prints one NAME=value per line; cmd's hidden `=C:=...` entries
    // have an empty name and are dropped.
    let env: Vec<(String, String)> = stdout
        .lines()
        .filter_map(|line| line.split_once('='))
        .filter(|(name, _)| !name.is_empty())
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect();
    // VsDevCmd.bat exits 0 having printed a usage banner when an argument is
    // wrong, so the two variables the compiler cannot work without are what
    // decides whether the batch did its job.
    let has = |want: &str| env.iter().any(|(name, _)| name.eq_ignore_ascii_case(want));
    if !has("INCLUDE") || !has("LIB") {
        return Err(format!(
            "cmd.exe {line} set no INCLUDE/LIB ({} variables):\n{}",
            env.len(),
            banner(&stdout)
        ));
    }
    Ok(env)
}

/// Decodes what a `cmd.exe /u` pipe carried.
///
/// cmd writes its own output as UTF-16LE under `/u`, while a program it ran
/// writes whatever bytes it likes into the same pipe. Text from cmd is
/// mostly ASCII, whose UTF-16LE encoding puts a zero in every second byte
/// and which no UTF-8 text contains at all, so the two are told apart by
/// counting those.
fn decode_console(bytes: &[u8]) -> String {
    let zeros = bytes.iter().skip(1).step_by(2).filter(|&&b| b == 0).count();
    if bytes.len() >= 2 && zeros * 4 >= bytes.len() {
        let units: Vec<u16> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect();
        return String::from_utf16_lossy(&units);
    }
    String::from_utf8_lossy(bytes).into_owned()
}

/// What a batch printed, without the environment `set` dumped after it.
///
/// The reason string reaches a CI log, and the environment there holds
/// tokens; only the lines that are not `NAME=value` say what went wrong.
fn banner(stdout: &str) -> String {
    stdout
        .lines()
        .filter(|line| line.split_once('=').is_none_or(|(name, _)| name.is_empty()))
        .take(20)
        .collect::<Vec<_>>()
        .join("\n")
}
