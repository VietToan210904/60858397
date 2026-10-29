param(
    [ValidateSet('core','retrieval','generation','assistance','3d')][string]$Pair='core',
    [switch]$InstallCuda,
    [switch]$ApproveLargeDownload
)
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$venv=Join-Path (Get-Location) ".venv-$Pair"
if (-not (Test-Path -LiteralPath "$venv\Scripts\python.exe")) {
    python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
}
$python=Join-Path $venv 'Scripts\python.exe'
if ($InstallCuda) {
    if (-not $ApproveLargeDownload) { throw 'CUDA plus model acquisition may exceed 5 GB. Obtain approval, then use -ApproveLargeDownload.' }
    & $python -m pip install 'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'CUDA wheel installation failed' }
}
& $python -m pip install -r "requirements-$Pair.txt"
if ($LASTEXITCODE -ne 0) { throw 'Project dependencies failed to install' }
& $python -m stylesphere.cli doctor
exit $LASTEXITCODE
