param([string]$Python='python')
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH=Join-Path (Get-Location) 'src'
& $Python -m stylesphere.cli doctor
exit $LASTEXITCODE
