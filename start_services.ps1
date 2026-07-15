$ErrorActionPreference = "Stop"

$repositoryRoot = $PSScriptRoot
$pythonPath = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
$frontendPath = Join-Path $repositoryRoot "frontend"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python virtual environment was not found: $pythonPath"
}

if (-not (Test-Path -LiteralPath (Join-Path $frontendPath "node_modules"))) {
    throw "Frontend dependencies are missing. Run npm install in frontend first."
}

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
    throw "$Name did not become ready within $TimeoutSeconds seconds: $Uri"
}

$apiUri = "http://127.0.0.1:8000/health"
$frontendUri = "http://127.0.0.1:5173/"

if (Test-LocalEndpoint -Uri $apiUri) {
    Write-Host "API is already running at http://127.0.0.1:8000"
}
elseif (Test-LocalPort -Port 8000) {
    throw "Port 8000 is occupied, but the expected API health endpoint is unavailable."
}
else {
    $apiCommand = "Set-Location -LiteralPath '$repositoryRoot'; & '$pythonPath' -m uvicorn api.main:app --host 127.0.0.1 --port 8000"
    Start-Process powershell.exe -ArgumentList @(
        "-NoProfile",
        "-Command",
        $apiCommand
    ) -WindowStyle Hidden
    Write-Host "Starting API at http://127.0.0.1:8000"
}

if (Test-LocalEndpoint -Uri $frontendUri) {
    Write-Host "Frontend is already running at http://127.0.0.1:5173"
}
elseif (Test-LocalPort -Port 5173) {
    throw "Port 5173 is occupied, but the expected frontend page is unavailable."
}
else {
    $frontendCommand = "Set-Location -LiteralPath '$frontendPath'; npm run dev"
    Start-Process powershell.exe -ArgumentList @(
        "-NoProfile",
        "-Command",
        $frontendCommand
    ) -WindowStyle Hidden
    Write-Host "Starting frontend at http://127.0.0.1:5173"
}

Wait-LocalEndpoint -Name "API" -Uri $apiUri
Wait-LocalEndpoint -Name "Frontend" -Uri $frontendUri
Start-Process $frontendUri
