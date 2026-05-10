param([string]$Python='python',[string]$Seed='42',[string]$Benchmark='codet_m4')
$ErrorActionPreference='Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $repoRoot
$env:FS_SEED=$Seed; $env:FS_SWEEP_KS=''; $env:FS_SWEEP_FRACS='0.10'; $env:FS_BENCHMARK=$Benchmark
& $Python 'Exp_FewShot/testing/exp_fs_baseline_unixcoder.py'
