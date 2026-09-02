#Requires -Version 7
<#
.SYNOPSIS
  Rewrite an LCOV report's SF: records to repo-relative POSIX paths.

.DESCRIPTION
  cargo-llvm-cov on Windows writes absolute SF: records with backslashes
  (D:\a\hal\hal\crates\image\src\lib.rs). SonarCloud resolves SF: against
  the scanner's own checkout, so every lane's report must agree on
  repo-relative forward-slash paths (crates/image/src/lib.rs) — the Unix
  lanes get the same treatment from a sed in the sonarcloud job. Records
  outside the root (registry sources that slipped past
  --ignore-filename-regex) are left untouched. The file is re-emitted
  LF-terminated so the Linux scanner never sees a stray CR.

.PARAMETER Lcov
  The LCOV file to rewrite in place.
.PARAMETER Root
  Repository root the SF: paths are made relative to (default: the parent
  of this script's directory, i.e. the workspace root).

.EXAMPLE
  pwsh scripts/normalize-lcov-paths.ps1 target/coverage_rust.lcov
#>
param(
    [Parameter(Mandatory)][string]$Lcov,
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)
$ErrorActionPreference = 'Stop'

$prefix = ($Root -replace '\\', '/').TrimEnd('/') + '/'
$rewritten = 0
$lines = @([System.IO.File]::ReadAllLines($Lcov) | ForEach-Object {
    if ($_.StartsWith('SF:')) {
        $p = $_.Substring(3) -replace '\\', '/'
        if ($p.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            $p = $p.Substring($prefix.Length)
            $rewritten++
        }
        "SF:$p"
    } else {
        $_
    }
})
[System.IO.File]::WriteAllText($Lcov, (($lines -join "`n") + "`n"))
Write-Host "[normalize-lcov-paths] ${Lcov}: $rewritten SF: records made relative to $prefix"
