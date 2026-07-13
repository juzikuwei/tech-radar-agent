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

function Test-LocalPort {
    param([int]$Port)

    return [bool](
        Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    )
}

if (Test-LocalPort -Port 8000) {
    Write-Host "API is already running at http://127.0.0.1:8000"
}
else {
    $apiCommand = "Set-Location -LiteralPath '$repositoryRoot'; & '$pythonPath' -m uvicorn api.main:app --host 127.0.0.1 --port 8000"
    Start-Process powershell.exe -ArgumentList @(
        "-NoExit",
        "-NoProfile",
        "-Command",
        $apiCommand
    )
    Write-Host "Starting API at http://127.0.0.1:8000"
}

if (Test-LocalPort -Port 5173) {
    Write-Host "Frontend is already running at http://127.0.0.1:5173"
}
else {
    $frontendCommand = "Set-Location -LiteralPath '$frontendPath'; npm run dev -- --host 127.0.0.1 --port 5173"
    Start-Process powershell.exe -ArgumentList @(
        "-NoExit",
        "-NoProfile",
        "-Command",
        $frontendCommand
    )
    Write-Host "Starting frontend at http://127.0.0.1:5173"
}

Start-Sleep -Seconds 2
Start-Process "http://127.0.0.1:5173"
