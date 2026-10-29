# Launch interactively on the user's computer. Never pass a token as an argument,
# save a transcript, or copy credentials into the project.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$hfCommand = (Get-Command hf -ErrorAction Stop).Source
$pythonPath = Join-Path $projectRoot '.venv-retrieval\Scripts\python.exe'
$checkScript = Join-Path $PSScriptRoot 'check_gated_access.py'

$Host.UI.RawUI.WindowTitle = 'StyleSphere - Private Hugging Face Login'
Write-Host 'Use a fresh READ token from https://huggingface.co/settings/tokens'
Write-Host 'Enter the token only in the official hidden prompt below, never in chat.'
Write-Host 'This saves it in the Hugging Face user cache, outside the project.'
Write-Host 'If asked to add the token as a Git credential, enter N.'
Write-Host ''
& $hfCommand auth login --force --no-add-to-git-credential
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Login did not succeed. No model downloads have started.'
    return
}
Write-Host ''
Write-Host 'Checking the two model gates without printing your identity or token...'
& $pythonPath $checkScript
if ($LASTEXITCODE -eq 0) {
    Write-Host 'Both model gates are accessible. Return to Codex to continue.'
} else {
    Write-Host 'Login succeeded, but at least one model gate remains inaccessible.'
    Write-Host 'Ensure the token belongs to the same account that accepted both model licences.'
}
