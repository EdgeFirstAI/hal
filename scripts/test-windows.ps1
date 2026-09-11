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
  otherwise). For CI runners and machines without a GPU. Also suppresses the
  -RequireCuda auto-detection below: WARP has no CUDA device, so a host
  signal must not require CUDA coverage of it.
.PARAMETER RequireGl
  Set HAL_TEST_REQUIRE_GL=1 so a GL backend that fails to come up fails the
  run (the `gl_backend_available_canary` test) instead of silently skipping.
.PARAMETER RequireCuda
  Set HAL_TEST_REQUIRE_CUDA=1 so the D3D11 CUDA interop tests in
  crates/tensor/tests/d3d11_tensor.rs fail loudly, naming the test and the
  reason, instead of reporting a skip for `no CUDA runtime` when the probe in
  crates/tensor/src/cuda.rs cannot load a runtime. The WARP-adapter skip
  (no CUDA device on the software adapter) is correct by design and is
  unaffected: those tests check the adapter before the gate.

  Auto-set when the *host* looks like an NVIDIA box -- `nvidia-smi` on PATH
  and exiting zero, or CUDA_PATH set -- so a real NVIDIA box requires CUDA
  coverage even without passing this switch. Note what those two signals do
  and do not say: they establish that an NVIDIA driver or toolkit is
  installed, not that the D3D11 device will be created on that adapter.
  The adapter is chosen by EDGEFIRST_D3D11_ADAPTER (or its
  EDGEFIRST_ANGLE_ADAPTER alias), and unset means DXGI adapter 0, which on a
  hybrid laptop can be the integrated GPU -- see
  crates/tensor/src/d3d11/adapter.rs. So auto-detection is suppressed when
  WARP is the selected adapter, by -Warp or by either variable inherited
  from the shell; on a hybrid box where adapter 0 is not the NVIDIA one,
  name it (EDGEFIRST_D3D11_ADAPTER=discrete, or a description substring such
  as 'RTX 3070') or pass -NoRequireCuda. Passing -RequireCuda explicitly
  always arms the gate, whatever the adapter.
.PARAMETER NoRequireCuda
  Suppress the NVIDIA-adapter auto-detection for -RequireCuda above, leaving
  HAL_TEST_REQUIRE_CUDA unset even when an NVIDIA adapter is present. Clears
  the variable if the calling shell already exported it, so this switch
  disarms the gate whatever the environment says.
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
  Everything after the named switches is passed to `cargo nextest run`
  (e.g. `-p edgefirst-image -E 'test(~pbo)'`) via the automatic `$args`.
  Not an advanced script (no [CmdletBinding()] and no [Parameter()]
  attributes): either would add PowerShell's common parameters and make
  `-p` ambiguous with -ProgressAction/-PipelineVariable.

.EXAMPLE
  pwsh scripts/test-windows.ps1 -RequireGl                 # real GPU
  pwsh scripts/test-windows.ps1 -Warp -RequireGl -p edgefirst-image
  pwsh scripts/test-windows.ps1 -Coverage -Warp -RequireGl -p edgefirst-image --profile ci
  pwsh scripts/test-windows.ps1 -RequireGl -RequireCuda    # real GPU + CUDA coverage required
#>
param(
    [switch]$Warp,
    [switch]$RequireGl,
    [switch]$RequireCuda,
    [switch]$NoRequireCuda,
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

# Which adapter the D3D11 device will actually be created on, resolved by the
# same rule as crates/tensor/src/d3d11/adapter.rs: EDGEFIRST_D3D11_ADAPTER is
# the current name and EDGEFIRST_ANGLE_ADAPTER the alias it wins over. -Warp
# set the alias above, so this catches both the switch and a value inherited
# from the calling shell.
$adapterSel = if ($env:EDGEFIRST_D3D11_ADAPTER) {
    $env:EDGEFIRST_D3D11_ADAPTER
} else {
    $env:EDGEFIRST_ANGLE_ADAPTER
}
$warpSelected = (
    $null -ne $adapterSel -and
    $adapterSel.Trim().ToLowerInvariant() -in @('warp', 'software')
)

# An NVIDIA host without -RequireCuda used to let the D3D11 CUDA interop tests
# skip silently -- the same vacuity -RequireGl closes for GL. Detect it the
# simplest robust way available from PowerShell: `nvidia-smi` on PATH and
# exiting zero (works even if the box has no CUDA *toolkit*, just the driver),
# or CUDA_PATH set (the toolkit installer's own marker). Both describe the
# host; neither says which adapter the D3D11 device lands on, so do not arm
# the gate when WARP is the selected adapter -- an NVIDIA box run under -Warp
# is exactly the combination where a host signal would require CUDA coverage
# of a device that can never have it.
if (-not $NoRequireCuda -and -not $RequireCuda -and -not $warpSelected) {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction Ignore
    $nvidiaPresent = $false
    if ($nvidiaSmi) {
        & nvidia-smi | Out-Null
        if ($LASTEXITCODE -eq 0) { $nvidiaPresent = $true }
    }
    if ($env:CUDA_PATH) { $nvidiaPresent = $true }
    if ($nvidiaPresent) {
        Write-Host 'NVIDIA driver or toolkit present: requiring CUDA coverage'
        $RequireCuda = $true
    }
}
# The switches decide, not the calling shell. HAL_TEST_REQUIRE_CUDA is
# exported to the nextest child, so an inherited value would otherwise
# survive -NoRequireCuda and contradict the banner below: clear it when the
# gate is off, and treat an inherited `1` as arming when nothing suppresses
# it so the banner reports what the tests will actually see.
if (-not $NoRequireCuda -and $env:HAL_TEST_REQUIRE_CUDA -eq '1') { $RequireCuda = $true }
if ($warpSelected -and -not $RequireCuda) {
    Write-Host 'WARP adapter selected: not auto-requiring CUDA coverage'
}
$env:HAL_TEST_REQUIRE_CUDA = if ($RequireCuda) { '1' } else { $null }

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

Write-Host "[test-windows] EDGEFIRST_ANGLE_PATH=$env:EDGEFIRST_ANGLE_PATH adapter=$($env:EDGEFIRST_ANGLE_ADAPTER ?? 'default') require_gl=$([bool]$RequireGl) require_cuda=$([bool]$RequireCuda) coverage=$([bool]$Coverage)"
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
