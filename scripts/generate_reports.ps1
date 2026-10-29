param([string]$Python='python')
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH=Join-Path (Get-Location) 'src'
& $Python -m stylesphere.cli report
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m stylesphere.cli verify
exit $LASTEXITCODE
