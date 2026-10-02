# Start Audio Deepfake Detector

$env:PYTHONPATH="."
Write-Host "Starting Backend (FastAPI)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit -Command cd '$PWD'; `$env:PYTHONPATH='.'; .\backend\venv_tf\Scripts\python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

Write-Host "Starting Frontend (React+Vite)..." -ForegroundColor Magenta
Start-Process powershell -ArgumentList "-NoExit -Command cd '$PWD\frontend'; npm run dev"

Write-Host "----------------------------------------------------"
Write-Host "Backend running at: http://localhost:8000"
Write-Host "Frontend running at: http://localhost:3000"
Write-Host "----------------------------------------------------"
Write-Host "Make sure MongoDB is running on localhost:27017"
