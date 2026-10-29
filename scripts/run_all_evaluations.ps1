param([string]$Config='')
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH=Join-Path (Get-Location) 'src'
$partial=$false
foreach ($pair in @('retrieval','assistance','generation','reconstruction')) {
    $suffix=if ($pair -eq 'reconstruction') { '3d' } else { $pair }
    $python=".venv-$suffix\Scripts\python.exe"
    if ($pair -eq 'reconstruction' -and -not (Test-Path -LiteralPath $python)) { $python='.venv-retrieval\Scripts\python.exe' }
    if (-not (Test-Path -LiteralPath $python)) { Write-Warning "Missing isolated interpreter $python"; $partial=$true; continue }
    $cliArgs=@('-m','stylesphere.cli','evaluate','--pair',$pair)
    if ($Config) { $cliArgs+=@('--config',$Config) }
    & $python @cliArgs
    if ($LASTEXITCODE -ne 0) { $partial=$true }
}
if ($partial) { exit 2 }
exit 0
