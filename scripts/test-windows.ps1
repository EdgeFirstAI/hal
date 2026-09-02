#Requires -Version 7
<#
.SYNOPSIS
  test-windows.ps1 — run the Rust test suite on Windows with the ANGLE/D3D11
  GL backend reachable (the role scripts/test-macos.sh plays on macOS).

.DESCRIPTION
  Points EDGEFIRST_ANGLE_PATH at the ANGLE DLLs fetched by
  `bash scripts/fetch-angle.sh` (target\angle\windows-x64\bin) unless it is
  already set, optionally forces the WARP software adapter for GPU-less
  hosts, and runs `cargo nextest run --workspace -j 1` with the same crate
  exclusions as the macOS lane. `-j 1` matters: ANGLE takes the Full GL
  serialization policy, so GL tests must not overlap (see
  crates/image/ARCHITECTURE.md § GL Concurrency Model).

  Run from a "Developer PowerShell for VS" (or any shell where cargo can
  find MSVC link.exe); do not run cargo from Git Bash, whose
  /usr/bin/link.exe shadows the MSVC linker.

.PARAMETER Warp
  Use ANGLE's D3D11 WARP (software) adapter: sets EDGEFIRST_ANGLE_ADAPTER=warp
  and EDGEFIRST_ALLOW_SOFTWARE_GL=1 (the backend rejects software renderers
  otherwise). For CI runners and machines without a GPU.
.PARAMETER RequireGl
  Set HAL_TEST_REQUIRE_GL=1 so a GL backend that fails to come up fails the
  run (the `gl_backend_available_canary` test) instead of silently skipping.
.PARAMETER Release
  Build tests with --release.
.PARAMETER Coverage
  Run under cargo-llvm-cov (`cargo llvm-cov nextest --no-report`) so the
  instrumented run leaves its profraw under target\llvm-cov-target. Several
  passes accumulate (the CI lane runs the no-ANGLE gating pass and the WARP
  GL pass this way), then `cargo llvm-cov report --lcov` merges them and
  scripts/normalize-lcov-paths.ps1 makes the SF: paths repo-relative for
  SonarCloud. Needs `rustup component add llvm-tools-preview` and
  `cargo install cargo-llvm-cov --locked`.
.NOTES
  Everything after the three switches is passed to `cargo nextest run`
  (e.g. `-p edgefirst-image -E 'test(~pbo)'`) via the automatic `$args`.
  Not an advanced script (no [CmdletBinding()] and no [Parameter()]
  attributes): either would add PowerShell's common parameters and make
  `-p` ambiguous with -ProgressAction/-PipelineVariable.

.EXAMPLE
  pwsh scripts/test-windows.ps1 -RequireGl                 # real GPU
  pwsh scripts/test-windows.ps1 -Warp -RequireGl -p edgefirst-image
  pwsh scripts/test-windows.ps1 -Coverage -Warp -RequireGl -p edgefirst-image --profile ci
#>
param(
    [switch]$Warp,
    [switch]$RequireGl,
    [switch]$Release,
    [switch]$Coverage
)
$NextestArgs = @($args)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

if (-not $env:EDGEFIRST_ANGLE_PATH) {
    $env:EDGEFIRST_ANGLE_PATH = Join-Path $root 'target\angle\windows-x64\bin'
}
if (-not (Test-Path (Join-Path $env:EDGEFIRST_ANGLE_PATH 'libEGL.dll'))) {
    Write-Warning "no libEGL.dll under EDGEFIRST_ANGLE_PATH=$env:EDGEFIRST_ANGLE_PATH — run 'bash scripts/fetch-angle.sh' first (GL tests will self-skip$(if ($RequireGl) { ', and -RequireGl will FAIL the canary' }))"
}
if ($Warp) {
    $env:EDGEFIRST_ANGLE_ADAPTER = 'warp'
    $env:EDGEFIRST_ALLOW_SOFTWARE_GL = '1'
}
if ($RequireGl) { $env:HAL_TEST_REQUIRE_GL = '1' }
if (-not (Get-Command cargo -ErrorAction Ignore)) {
    throw 'cargo not found on PATH'
}
if (-not (Get-Command cargo-nextest -ErrorAction Ignore)) {
    throw 'cargo-nextest not found on PATH (cargo install cargo-nextest --locked)'
}
if ($Coverage -and -not (Get-Command cargo-llvm-cov -ErrorAction Ignore)) {
    throw 'cargo-llvm-cov not found on PATH (cargo install cargo-llvm-cov --locked; rustup component add llvm-tools-preview)'
}

# Same exclusions as the macOS lane: gpu-probe is Linux-only (gbm/nix);
# the python-* crates cannot join a --workspace build (edgefirst-tensor's
# static/dynamic feature exclusion) and are exercised through maturin.
$exclude = @(
    '--exclude', 'gpu-probe',
    '--exclude', 'edgefirst-bench',
    '--exclude', 'edgefirst-python-tensor',
    '--exclude', 'edgefirst-python-codec',
    '--exclude', 'edgefirst-python-image',
    '--exclude', 'edgefirst-python-decoder',
    '--exclude', 'edgefirst-python-tracker',
    '--exclude', 'edgefirst-python-common'
)
$profile = @(); if ($Release) { $profile = @('--release') }
# `--workspace` would override an explicit `-p <crate>` (cargo unions the
# selection), so only default to the whole workspace when no package is named.
$scope = @('--workspace') + $exclude
if ($NextestArgs -match '^(-p|--package)(=.*)?$') { $scope = @() }

Write-Host "[test-windows] EDGEFIRST_ANGLE_PATH=$env:EDGEFIRST_ANGLE_PATH adapter=$($env:EDGEFIRST_ANGLE_ADAPTER ?? 'default') require_gl=$([bool]$RequireGl) coverage=$([bool]$Coverage)"
Set-Location $root
if ($Coverage) {
    # --no-report leaves the profraw under target\llvm-cov-target so several
    # passes merge into one later `cargo llvm-cov report` (as the CI lane
    # does: no-ANGLE gating pass + WARP GL pass -> one LCOV). Everything
    # after --no-report is forwarded to nextest, `--profile <name>` included.
    & cargo llvm-cov nextest --no-report @scope @profile -j 1 @NextestArgs
} else {
    & cargo nextest run @scope @profile -j 1 @NextestArgs
}
exit $LASTEXITCODE
