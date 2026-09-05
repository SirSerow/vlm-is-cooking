$ErrorActionPreference = 'Stop'
$reviewRoot = Split-Path -Parent $PSScriptRoot
$reviewPython = Join-Path $reviewRoot '.venv-training/Scripts/python.exe'
try {
    $reviewResponse = Invoke-RestMethod 'http://127.0.0.1:8877/api/status' -TimeoutSec 3
    if ($null -ne $reviewResponse.total) {
        Write-Output 'Kitchen Lab is already running on port 8877.'
        exit 0
    }
} catch { }
Start-Process -FilePath $reviewPython -ArgumentList 'scripts/review_server.py' -WorkingDirectory $reviewRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $reviewRoot 'outputs/review-server.log') -RedirectStandardError (Join-Path $reviewRoot 'outputs/review-server-error.log')
Write-Output 'Started Kitchen Lab at http://127.0.0.1:8877'
