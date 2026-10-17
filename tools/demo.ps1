# Demo: replay server (pipeline per scene, seekable) + React FE.
# Usage: powershell -File tools/demo.ps1 [-Python <python.exe>]
#   Web: chon "Replay", chon scene, Space = play/pause, <- -> = tung frame, G = ground truth.
#   Ghi MP4 du phong (OpenCV, khong can web): python -m src.app --source phenikaa --root data/phenikaa
#       --scene "COMPLEX CITY" --cache cache --model yolo11m --renderer src.web_render:render --record results/replay_city.mp4
param([string]$Python = "")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
if (-not $Python) {
  $Python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
}
if (-not (Test-Path "frontend\node_modules")) { Push-Location frontend; npm install --no-audit --no-fund; Pop-Location }

Write-Host "1/2 replay server http://127.0.0.1:8765 (chay pipeline cho moi scene trong nen, ~35 s)"
$server = Start-Process -PassThru -NoNewWindow $Python "-X utf8 -m src.replay_server"
Write-Host "2/2 FE (Vite)"
$fe = Start-Process -PassThru -WindowStyle Minimized -WorkingDirectory "frontend" "npm.cmd" "run dev -- --port 5173 --strictPort"
Start-Sleep -Seconds 4
Start-Process "http://localhost:5173/"
Write-Host "Mo http://localhost:5173/ . Nhan Enter de tat."
[void](Read-Host)
Stop-Process -Id $server.Id -ErrorAction SilentlyContinue
Stop-Process -Id $fe.Id -ErrorAction SilentlyContinue
