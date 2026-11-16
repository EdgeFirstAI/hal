#Requires -Version 7
<#
.SYNOPSIS
  Rewrite an LCOV report's SF: records to repo-relative POSIX paths.

.DESCRIPTION
  Rewrites absolute SF: records under the repository root to repo-relative
  forward-slash paths, leaves records outside the root untouched, and writes
  the file LF-terminated.

.PARAMETER Lcov
  The LCOV file to rewrite in place.
.PARAMETER Root
  Repository root the SF: paths are made relative to (default: the parent
  of this script's directory).

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
