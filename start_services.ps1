$ErrorActionPreference = "Stop"

$repositoryRoot = $PSScriptRoot
$pythonPath = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
$frontendPath = Join-Path $repositoryRoot "frontend"
$viteScriptPath = Join-Path $frontendPath "node_modules\vite\bin\vite.js"

$runDirectory = Join-Path $repositoryRoot ".run"
$logsDirectory = Join-Path $repositoryRoot "logs"
$apiPidFile = Join-Path $runDirectory "api.pid"
$frontendPidFile = Join-Path $runDirectory "frontend.pid"
$apiLog = Join-Path $logsDirectory "api.log"
$apiErrorLog = Join-Path $logsDirectory "api.err.log"
$frontendLog = Join-Path $logsDirectory "frontend.log"
$frontendErrorLog = Join-Path $logsDirectory "frontend.err.log"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python virtual environment was not found: $pythonPath"
}

if (-not (Test-Path -LiteralPath (Join-Path $frontendPath "node_modules"))) {
    throw "Frontend dependencies are missing. Run npm install in frontend first."
}

if (-not (Test-Path -LiteralPath $viteScriptPath)) {
    throw "Vite entry script was not found: $viteScriptPath"
}

$nodeCommand = Get-Command node -ErrorAction SilentlyContinue
if ($null -eq $nodeCommand) {
    throw "node was not found on PATH. Install Node.js first."
}

New-Item -ItemType Directory -Force -Path $runDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $logsDirectory | Out-Null

function Test-LocalEndpoint {
    param([string]$Uri)

    try {
        $response = Invoke-WebRequest `
            -Uri $Uri `
            -UseBasicParsing `
            -TimeoutSec 2
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

function Test-LocalPort {
    param([int]$Port)

    return [bool](
        Get-NetTCPConnection `
            -State Listen `
            -LocalPort $Port `
            -ErrorAction SilentlyContinue
    )
}

function Wait-LocalEndpoint {
    param(
        [string]$Name,
        [string]$Uri,
        [string]$LogHint,
        [int]$TimeoutSeconds = 60
    )

    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    while ($timer.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        if (Test-LocalEndpoint -Uri $Uri) {
            Write-Host "$Name is ready at $Uri"
            return
        }
        Start-Sleep -Milliseconds 500
    }
    throw "$Name did not become ready within $TimeoutSeconds seconds: $Uri. Check logs: $LogHint"
}

function Stop-StartedProcess {
    param(
        [System.Diagnostics.Process]$Process,
        [string]$PidFile
    )

    if ($null -ne $Process) {
        try {
            if (-not $Process.HasExited) {
                Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
            }
        }
        catch {
            # The process already exited between the check and the stop call.
        }
        if (Test-Path -LiteralPath $PidFile) {
            Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
        }
    }
}

$apiUri = "http://127.0.0.1:8000/health"
$frontendUri = "http://127.0.0.1:5173/"

$apiProcess = $null
$frontendProcess = $null

try {
    if (Test-LocalEndpoint -Uri $apiUri) {
        Write-Host "API is already running at http://127.0.0.1:8000"
    }
    elseif (Test-LocalPort -Port 8000) {
        throw "Port 8000 is occupied, but the expected API health endpoint is unavailable."
    }
    else {
        $apiProcess = Start-Process -FilePath $pythonPath -ArgumentList @(
            "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"
        ) -WorkingDirectory $repositoryRoot `
            -WindowStyle Hidden `
            -PassThru `
            -RedirectStandardOutput $apiLog `
            -RedirectStandardError $apiErrorLog
        Set-Content -LiteralPath $apiPidFile -Value $apiProcess.Id
        Write-Host "Starting API at http://127.0.0.1:8000 (PID $($apiProcess.Id), logs: $apiLog, $apiErrorLog)"
    }

    if (Test-LocalEndpoint -Uri $frontendUri) {
        Write-Host "Frontend is already running at http://127.0.0.1:5173"
    }
    elseif (Test-LocalPort -Port 5173) {
        throw "Port 5173 is occupied, but the expected frontend page is unavailable."
    }
    else {
        $frontendProcess = Start-Process -FilePath $nodeCommand.Source -ArgumentList @(
            "`"$viteScriptPath`""
        ) -WorkingDirectory $frontendPath `
            -WindowStyle Hidden `
            -PassThru `
            -RedirectStandardOutput $frontendLog `
            -RedirectStandardError $frontendErrorLog
        Set-Content -LiteralPath $frontendPidFile -Value $frontendProcess.Id
        Write-Host "Starting frontend at http://127.0.0.1:5173 (PID $($frontendProcess.Id), logs: $frontendLog, $frontendErrorLog)"
    }

    Wait-LocalEndpoint -Name "API" -Uri $apiUri -LogHint "$apiLog, $apiErrorLog"
    Wait-LocalEndpoint -Name "Frontend" -Uri $frontendUri -LogHint "$frontendLog, $frontendErrorLog"
}
catch {
    Stop-StartedProcess -Process $apiProcess -PidFile $apiPidFile
    Stop-StartedProcess -Process $frontendProcess -PidFile $frontendPidFile
    throw
}

Start-Process $frontendUri
