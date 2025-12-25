$ErrorActionPreference = "Stop"
$CONDA_ENV_NAME = "pinns4preCastNodes"
$BACKEND_PORT = 8000
$FRONTEND_PORT = 3000

# Global cleanup
$script:BackendProcess = $null
$script:FrontendProcess = $null

function Cleanup {
    Write-Host "`n🛑 Stopping services..." -ForegroundColor Red
    if ($script:BackendProcess) { Stop-Process -Id $script:BackendProcess.Id -Force -ErrorAction SilentlyContinue }
    if ($script:FrontendProcess) { Stop-Process -Id $script:FrontendProcess.Id -Force -ErrorAction SilentlyContinue }
    Write-Host "✅ Services stopped" -ForegroundColor Green
}

# Trap Ctrl+C
Register-EngineEvent PowerShell.Exiting -Action { Cleanup }

Write-Host "🚀 Starting PINNs4PreCast (PowerShell Edition)..." -ForegroundColor Cyan

# Activate conda
conda activate $CONDA_ENV_NAME
Write-Host "✅ Conda: $CONDA_ENV_NAME activated" -ForegroundColor Green

# Check ports
Write-Host "🔍 Checking ports..." -ForegroundColor Yellow
$ports = @($BACKEND_PORT, $FRONTEND_PORT)
foreach ($port in $ports) {
    $listener = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($listener) {
        Write-Host "❌ Port $port in use by PID $($listener.OwningProcess)" -ForegroundColor Red
        exit 1
    }
}
Write-Host "✅ Ports available" -ForegroundColor Green

# === BACKEND ===
Write-Host "`n🐍 [1/2] Starting Backend..." -ForegroundColor Green
$script:BackendProcess = Start-Process -FilePath "uvicorn" -ArgumentList @(
    "src.api.main:app", 
    "--host", "0.0.0.0", 
    "--port", "$BACKEND_PORT", 
    "--log-level", "debug"
) -PassThru -NoNewWindow

Start-Sleep 3
if ($script:BackendProcess.HasExited) {
    Write-Host "❌ Backend failed (PID $($script:BackendProcess.Id))" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Backend PID: $($script:BackendProcess.Id)" -ForegroundColor Green

# === FRONTEND ===
Write-Host "`n⚛️ [2/2] Starting Frontend..." -ForegroundColor Green
if (-not (Test-Path "frontend")) {
    Write-Host "❌ frontend/ missing. Run: cd frontend && npm install" -ForegroundColor Red
    exit 1
}

$script:FrontendProcess = Start-Process -FilePath "powershell" -ArgumentList @(
    "-NoExit", 
    "-Command", 
    "cd frontend; npm run dev -- -p $FRONTEND_PORT"
) -PassThru -WindowStyle Normal

Start-Sleep 3
if ($script:FrontendProcess.HasExited) {
    Write-Host "❌ Frontend failed. Run: cd frontend && npm install && npm run dev" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Frontend PID: $($script:FrontendProcess.Id)" -ForegroundColor Green

Write-Host "`n🎉 PINNs4PreCast LIVE!" -ForegroundColor Cyan
Write-Host "   🐍 Backend:  http://localhost:$BACKEND_PORT/docs" -ForegroundColor Green
Write-Host "   ⚛️  Frontend: http://localhost:$FRONTEND_PORT" -ForegroundColor Green
Write-Host "   💥 Ctrl+C to stop everything" -ForegroundColor Yellow

# Wait forever (both processes run until Ctrl+C)
while ($true) {
    Start-Sleep 1
    if ($script:BackendProcess.HasExited) { Write-Host "⚠️ Backend exited!" -ForegroundColor Red; break }
    if ($script:FrontendProcess.HasExited) { Write-Host "⚠️ Frontend exited!" -ForegroundColor Yellow }
}
Cleanup
