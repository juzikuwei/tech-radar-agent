$ErrorActionPreference = "Stop"

$repositoryRoot = $PSScriptRoot
$runDirectory = Join-Path $repositoryRoot ".run"

function Stop-ServiceFromPidFile {
    param(
        [string]$Name,
        [string]$PidFile,
        [string[]]$ExpectedProcessNames
    )

    if (-not (Test-Path -LiteralPath $PidFile)) {
        Write-Host "$Name is not tracked (no pid file at $PidFile); nothing to stop."
        return
    }

    $rawPid = (Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    $processId = 0
    if (-not [int]::TryParse(("$rawPid").Trim(), [ref]$processId)) {
        Write-Host "$Name pid file is invalid; removing it."
        Remove-Item -LiteralPath $PidFile -Force
        return
    }

    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        Write-Host "$Name process $processId is not running; removing stale pid file."
    }
    elseif ($ExpectedProcessNames -and ($ExpectedProcessNames -notcontains $process.ProcessName)) {
        Write-Host "$Name pid $processId now belongs to '$($process.ProcessName)', not $($ExpectedProcessNames -join '/'); leaving it alone and removing the stale pid file."
    }
    else {
        Stop-Process -Id $processId -Force
        Write-Host "$Name process $processId stopped."
    }

    Remove-Item -LiteralPath $PidFile -Force
}

Stop-ServiceFromPidFile `
    -Name "API" `
    -PidFile (Join-Path $runDirectory "api.pid") `
    -ExpectedProcessNames @("python")

Stop-ServiceFromPidFile `
    -Name "Frontend" `
    -PidFile (Join-Path $runDirectory "frontend.pid") `
    -ExpectedProcessNames @("node")
