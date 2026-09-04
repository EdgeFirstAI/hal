# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Builds d3d11_probe.exe with the MSVC toolchain found through vswhere.
# Output goes to <repo>/target/d3d11-probe/. The CUDA headers are used for
# types only; every CUDA, ANGLE and DirectML entry point is loaded at run
# time, so nothing beyond the Windows SDK is linked.
#
#   pwsh crates/gpu-probe/windows/build.ps1            # build
#   pwsh crates/gpu-probe/windows/build.ps1 -Run       # build and run on the default adapter
#   pwsh crates/gpu-probe/windows/build.ps1 -Run -Warp # build and run on WARP
#   pwsh crates/gpu-probe/windows/build.ps1 -Run -ProbeArgs @("--only","s1,s5")
#
# One spelling for the probe's own arguments, matching README.md: this is an
# advanced script ([CmdletBinding()]), so PowerShell reads a bare `--` as an
# ambiguous parameter name rather than as an end-of-options marker.
#
# Invoke it as `pwsh <path>` as shown, not as `& ./build.ps1` or by
# dot-sourcing: `Launch-VsDevShell.ps1` below sets PATH, INCLUDE and LIB in
# the *calling* process and never restores them, so the MSVC environment
# would leak into the shell you ran it from. `pwsh <path>` spawns a child
# process, which takes the leak with it.
[CmdletBinding()]
param(
    [switch]$Run,
    [switch]$Warp,
    [string]$CudaInclude = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ProbeArgs = @()
)

$ErrorActionPreference = "Stop"
$src = $PSScriptRoot
$repo = (Resolve-Path (Join-Path $src "..\..\..")).Path
$out = Join-Path $repo "target\d3d11-probe"
New-Item -ItemType Directory -Force $out | Out-Null

if (-not $CudaInclude) {
    if ($env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH "include\cuda_d3d11_interop.h"))) {
        $CudaInclude = Join-Path $env:CUDA_PATH "include"
    } else {
        $candidates = Get-ChildItem "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending
        foreach ($c in $candidates) {
            if (Test-Path (Join-Path $c.FullName "include\cuda_d3d11_interop.h")) { $CudaInclude = Join-Path $c.FullName "include"; break }
        }
    }
}
if (-not $CudaInclude) { throw "CUDA toolkit headers not found; pass -CudaInclude <dir>" }

$angleInclude = Join-Path $repo "target\angle\windows-x64\include"
if (-not (Test-Path (Join-Path $angleInclude "EGL\eglext_angle.h"))) {
    throw "ANGLE headers not found at $angleInclude; run 'bash scripts/fetch-angle.sh' first"
}

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found; install Visual Studio 2022 or newer" }
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "no Visual Studio installation with the C++ toolset found" }
$devshell = Join-Path $vs "Common7\Tools\Launch-VsDevShell.ps1"
& $devshell -Arch amd64 -SkipAutomaticLocation | Out-Null

$sources = Get-ChildItem (Join-Path $src "*.cpp") | ForEach-Object { $_.FullName }
Push-Location $out
try {
    $clArgs = @("/nologo", "/std:c++20", "/EHsc", "/W3", "/O2", "/Zi", "/MT",
                "/D_CRT_SECURE_NO_WARNINGS", "/DUNICODE", "/D_UNICODE",
                "/I", $angleInclude, "/I", $CudaInclude, "/Fe:d3d11_probe.exe") +
              $sources +
              @("/link", "d3d11.lib", "dxgi.lib", "d3d12.lib", "user32.lib", "ole32.lib", "synchronization.lib")
    & cl @clArgs
    if ($LASTEXITCODE -ne 0) { throw "cl.exe failed ($LASTEXITCODE)" }
    Write-Host "built $out\d3d11_probe.exe"
} finally {
    Pop-Location
}

if ($Run) {
    $args2 = @()
    if ($Warp) { $args2 += "--warp" }
    $args2 += $ProbeArgs
    & (Join-Path $out "d3d11_probe.exe") @args2
    exit $LASTEXITCODE
}
