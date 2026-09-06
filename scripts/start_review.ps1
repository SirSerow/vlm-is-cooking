$ErrorActionPreference = 'Stop'
$reviewRoot = Split-Path -Parent $PSScriptRoot
$reviewPython = Join-Path $reviewRoot '.venv-training/Scripts/python.exe'
$reviewPort = 8877

function Get-LocalListeners {
    @(Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $reviewPort -State Listen -ErrorAction SilentlyContinue)
}

function Stop-KitchenLab {
    $processIds = @(
        Get-LocalListeners |
            Select-Object -ExpandProperty OwningProcess -Unique
    )
    $processIds += @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -match '^python(\.exe)?$' -and
                $_.CommandLine -match 'scripts[/\\]review_server\.py'
            } |
            Select-Object -ExpandProperty ProcessId
    )

    foreach ($processId in ($processIds | Select-Object -Unique)) {
        if ($processId -and $processId -gt 4) {
            Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        }
    }

    $deadline = (Get-Date).AddSeconds(5)
    do {
        if (-not (Get-LocalListeners)) {
            return
        }
        Start-Sleep -Milliseconds 200
    } while ((Get-Date) -lt $deadline)

    $leftover = Get-LocalListeners | ForEach-Object {
        $proc = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue
        '{0} ({1})' -f $_.OwningProcess, $(if ($proc) { $proc.ProcessName } else { 'unknown' })
    }
    throw "Port $reviewPort on 127.0.0.1 is still in use after stopping Kitchen Lab: $($leftover -join ', ')"
}

Stop-KitchenLab
Start-Process -FilePath $reviewPython -ArgumentList 'scripts/review_server.py' -WorkingDirectory $reviewRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $reviewRoot 'outputs/review-server.log') -RedirectStandardError (Join-Path $reviewRoot 'outputs/review-server-error.log')
Write-Output "Started Kitchen Lab at http://127.0.0.1:$reviewPort"