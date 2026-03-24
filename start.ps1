# Start Audio Deepfake Detector

$env:PYTHONPATH="."
Write-Host "Starting Backend (FastAPI)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit -Command cd '$PWD'; .\backend\venv\Scripts\python -m backend.main"

Write-Host "Starting Frontend (React+Vite)..." -ForegroundColor Magenta
Start-Process powershell -ArgumentList "-NoExit -Command cd '$PWD\frontend'; npm run dev"

Write-Host "----------------------------------------------------"
Write-Host "Backend running at: http://localhost:8000"
Write-Host "Frontend running at: http://localhost:3000"
Write-Host "----------------------------------------------------"
Write-Host "Make sure MongoDB is running on localhost:27017"
