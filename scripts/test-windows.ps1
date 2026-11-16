#Requires -Version 7
<#
.SYNOPSIS
  test-windows.ps1 — run the Rust test suite on Windows with the ANGLE/D3D11
  GL backend reachable.

.DESCRIPTION
  Points EDGEFIRST_ANGLE_PATH at the ANGLE DLLs fetched by
  `bash scripts/fetch-angle.sh` (target\angle\windows-x64\bin) unless it is
  already set, optionally selects the WARP software adapter, and runs
  `cargo nextest run --workspace -j 1` with the crate exclusions of the macOS
  lane.

  Run from a "Developer PowerShell for VS", or any shell where cargo finds
  MSVC link.exe.

.PARAMETER Warp
  Use ANGLE's D3D11 WARP (software) adapter: sets EDGEFIRST_ANGLE_ADAPTER=warp
  and EDGEFIRST_ALLOW_SOFTWARE_GL=1, and turns off the -RequireCuda
  auto-detection.
.PARAMETER RequireGl
  Set HAL_TEST_REQUIRE_GL=1, which fails the run when the GL backend does not
  come up.
.PARAMETER RequireCuda
  Set HAL_TEST_REQUIRE_CUDA=1, which fails the D3D11 CUDA interop tests when
  no CUDA runtime loads. Set automatically when `nvidia-smi` runs or CUDA_PATH
  is set, unless WARP is the selected adapter (-Warp, EDGEFIRST_D3D11_ADAPTER
  or EDGEFIRST_ANGLE_ADAPTER).
.PARAMETER NoRequireCuda
  Turn off the -RequireCuda auto-detection and clear HAL_TEST_REQUIRE_CUDA.
.PARAMETER Release
  Build tests with --release.
.PARAMETER Coverage
  Run under `cargo llvm-cov nextest --no-report`, leaving profraw under
  target\llvm-cov-target. Needs `rustup component add llvm-tools-preview` and
  `cargo install cargo-llvm-cov --locked`.
.NOTES
  Everything after the named switches is passed to `cargo nextest run`
  (e.g. `-p edgefirst-image -E 'test(~pbo)'`) through `$args`.

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

# The adapter the D3D11 device is created on: EDGEFIRST_D3D11_ADAPTER, or
# its alias EDGEFIRST_ANGLE_ADAPTER when that is unset.
$adapterSel = if ($env:EDGEFIRST_D3D11_ADAPTER) {
    $env:EDGEFIRST_D3D11_ADAPTER
} else {
    $env:EDGEFIRST_ANGLE_ADAPTER
}
$warpSelected = (
    $null -ne $adapterSel -and
    $adapterSel.Trim().ToLowerInvariant() -in @('warp', 'software')
)

# Requires CUDA coverage when `nvidia-smi` is on PATH and exits zero, or
# CUDA_PATH is set, unless WARP is the selected adapter.
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
# An inherited HAL_TEST_REQUIRE_CUDA=1 arms the gate unless -NoRequireCuda is
# given; the variable is cleared when the gate is off.
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

# Excludes gpu-probe, the bench crate and the python-* crates.
$exclude = @(
    '--exclude', 'gpu-probe',
    '--exclude', 'edgefirst-bench',
    '--exclude', 'edgefirst-python-tensor',
    '--exclude', 'edgefirst-python-codec',
    '--exclude', 'edgefirst-python-image',
    '--exclude', 'edgefirst-python-decoder',
    '--exclude', 'edgefirst-python-tracker',
    '--exclude', 'edgefirst-python-kernels',
    '--exclude', 'edgefirst-python-common'
)
$profile = @(); if ($Release) { $profile = @('--release') }
# Defaults to the whole workspace when no package is named.
$scope = @('--workspace') + $exclude
if ($NextestArgs -match '^(-p|--package)(=.*)?$') { $scope = @() }

Write-Host "[test-windows] EDGEFIRST_ANGLE_PATH=$env:EDGEFIRST_ANGLE_PATH adapter=$($env:EDGEFIRST_ANGLE_ADAPTER ?? 'default') require_gl=$([bool]$RequireGl) require_cuda=$([bool]$RequireCuda) coverage=$([bool]$Coverage)"
Set-Location $root
if ($Coverage) {
    # --no-report leaves the profraw under target\llvm-cov-target. Everything
    # after --no-report is forwarded to nextest.
    & cargo llvm-cov nextest --no-report @scope @profile -j 1 @NextestArgs
} else {
    & cargo nextest run @scope @profile -j 1 @NextestArgs
}
exit $LASTEXITCODE
