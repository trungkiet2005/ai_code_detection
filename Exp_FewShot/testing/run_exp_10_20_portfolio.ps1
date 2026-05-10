param(
    [string]$Python = 'python',
    [string]$Seed = '42',
    [string]$Fractions = '0.10,0.20',
    [string]$Benchmark = 'codet_m4'
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $repoRoot

Write-Host "[run] repoRoot=$repoRoot"
Write-Host "[run] Seed=$Seed Fractions=$Fractions Benchmark=$Benchmark"

$env:FS_SEED = $Seed
$env:FS_SWEEP_KS = ''
$env:FS_SWEEP_FRACS = $Fractions
$env:FS_BENCHMARK = $Benchmark

$jobs = @(
    'Exp_FewShot/testing/exp_fs_inline_hier_ntk.py',
    'Exp_FewShot/testing/exp_fs_baseline_unixcoder.py',
    'Exp_FewShot/testing/exp_fs_inline_baseline.py'
)

foreach ($job in $jobs) {
    Write-Host "`n[run] $job"
    & $Python $job
}

Write-Host "`n[run] aggregate_fs_results.py"
& $Python 'Exp_FewShot/aggregate_fs_results.py'

Write-Host "`n[done] Finished 10%/20% portfolio run."
