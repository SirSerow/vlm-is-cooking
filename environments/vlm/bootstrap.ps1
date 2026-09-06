param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$venvPython = Join-Path $projectRoot '.venv-vlm/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $Python -m venv (Join-Path $projectRoot '.venv-vlm')
    if ($LASTEXITCODE -ne 0) { throw 'Python venv creation failed; pass -Python with a Python 3.10+ executable.' }
}
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw 'Install Ollama for Windows from https://ollama.com/download/windows, then reopen PowerShell.'
}
try { Invoke-RestMethod 'http://127.0.0.1:11434/api/version' | Out-Null }
catch { throw 'Start Ollama (or run ollama serve in a separate terminal), then rerun bootstrap.' }
& ollama pull qwen3-vl:2b-instruct
if ($LASTEXITCODE -ne 0) { throw 'Model download failed.' }
& $venvPython -m unittest discover -s (Join-Path $projectRoot 'scripts') -p test_vlm_smoke.py -v
if ($LASTEXITCODE -ne 0) { throw 'VLM validation tests failed.' }
Write-Host "Ready: $venvPython"
