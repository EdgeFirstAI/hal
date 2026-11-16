// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//! MSVC toolchain discovery for the C tests on Windows.
//!
//! This module finds a working `cl.exe` once per test process: the current
//! environment when it is a developer shell, otherwise the newest Visual
//! Studio with the C++ toolset (through vswhere), whose `VsDevCmd.bat`
//! supplies PATH, INCLUDE and LIB.
//!
//! Discovery ends with a trivial program compiled and linked. Every failure
//! carries the command line and the compiler output in its reason string.
//!
//! Discovery also sets the process error mode to
//! `SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX`, which child processes
//! inherit: a crashing C program exits with a code instead of raising a
//! dialog.
//!
//! Test support only, `#[path]`-included by each modular C-API leaf's test
//! module.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

/// System libraries a leaf's Rust staticlib needs on top of its own archive.
pub const SYSTEM_LIBS: &[&str] = &[
    "legacy_stdio_definitions.lib",
    "kernel32.lib",
    "ntdll.lib",
    "userenv.lib",
    "ws2_32.lib",
    "dbghelp.lib",
];

/// Environment variables removed from every compiler invocation: `cl.exe`
/// prepends `CL` and appends `_CL_` to its command line.
const INJECTED_FLAG_VARS: [&str; 2] = ["CL", "_CL_"];

// SAFETY: kernel32 exports these with the documented signatures.
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

/// Report a skipped check by writing `WARNING: SKIPPED: {reason}` directly
/// to stderr.
pub fn skip(reason: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "WARNING: SKIPPED: {reason}");
}

/// The toolchain for this process, located once, or the reason there is
/// none.
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
    /// A per-process directory for this process's compiler output.
    /// [`sweep_scratch`] removes the directories of processes that have
    /// exited.
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
    /// When an environment was captured from `VsDevCmd.bat`, the caller's is
    /// cleared and replaced by it; otherwise the caller's is inherited. The
    /// variables in [`INJECTED_FLAG_VARS`] are removed in both cases.
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

    /// Parse `src` without generating code, with `/W4 /WX`. `flags` carries
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

    /// Compile `src` with `/MD` and link it with `libs` and [`SYSTEM_LIBS`]
    /// into `exe`.
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

    /// Names `binary` exports, from `dumpbin /exports`. `dumpbin.exe` is
    /// taken from the directory `cl.exe` is in.
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

/// Run `cmd` and fold its command line, stdout and stderr into one `stderr`.
///
/// A tool that cannot be spawned returns a failed `Output`.
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
    // Probe a developer shell's `cl.exe` first; a failure falls through to
    // vswhere.
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

/// Sets the process error mode so a crashing child exits with a code
/// instead of raising an error dialog. Children inherit the mode.
fn quiet_crash_dialogs() {
    // SAFETY: `SetErrorMode` has no preconditions.
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

/// Removes the scratch directories of test processes that have exited. The
/// current process's directory is kept.
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
        // A removal failure is ignored.
        let _ = std::fs::remove_dir_all(entry.path());
    }
}

/// Whether a process with this id is running. Only an id the kernel reports
/// as naming no process counts as gone.
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

/// Compile and link a trivial program with this toolchain. The error carries
/// the command line and everything the compiler printed.
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

/// vswhere's install location under the 32-bit program files, looked up
/// through `ProgramFiles(x86)`, `ProgramFiles` and the literal path.
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
    // Passed verbatim, under cmd's quoting rules. `/u` makes cmd write
    // UTF-16; `/s` strips the outer quotes and runs the quoted batch path
    // with its arguments; `set` then prints the environment the batch left.
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
    // The environment must carry INCLUDE and LIB.
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

/// Decodes what a `cmd.exe /u` pipe carried: UTF-16LE when zero bytes at odd
/// offsets make up at least a quarter of the input, UTF-8 otherwise.
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

/// What a batch printed, without the `NAME=value` lines `set` dumped after
/// it, limited to 20 lines.
fn banner(stdout: &str) -> String {
    stdout
        .lines()
        .filter(|line| line.split_once('=').is_none_or(|(name, _)| name.is_empty()))
        .take(20)
        .collect::<Vec<_>>()
        .join("\n")
}
