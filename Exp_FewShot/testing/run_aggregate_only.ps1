param([string]$Python='python')
$ErrorActionPreference='Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $repoRoot
& $Python 'Exp_FewShot/aggregate_fs_results.py'
