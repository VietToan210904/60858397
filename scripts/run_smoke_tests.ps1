param([ValidateSet('retrieval','generation','assistance','reconstruction')][string]$Pair='retrieval', [string]$Python='', [string]$Config='')
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH=Join-Path (Get-Location) 'src'
if (-not $Python) {
    $suffix=if ($Pair -eq 'reconstruction') { '3d' } else { $Pair }
    $Python=".venv-$suffix\Scripts\python.exe"
    if ($Pair -eq 'reconstruction' -and -not (Test-Path -LiteralPath $Python)) { $Python='.venv-retrieval\Scripts\python.exe' }
}
$cliArgs=@('-m','stylesphere.cli','smoke','--pair',$Pair)
if ($Config) { $cliArgs+=@('--config',$Config) }
& $Python @cliArgs
exit $LASTEXITCODE
