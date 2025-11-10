# 모든 서버를 PowerShell에서 실행하는 스크립트
# 사용법: .\start_all_servers.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "모든 서버 시작 중..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$basePath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $basePath

# 1. 메인 백엔드 서버 (포트 8000)
Write-Host "[1/3] 메인 백엔드 서버 시작 (포트 8000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$basePath'; uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
Start-Sleep -Seconds 3

# 2. 이미지 보정 서버 (포트 8003)
Write-Host "[2/3] 이미지 보정 서버 시작 (포트 8003)..." -ForegroundColor Yellow
$enhancementPath = Join-Path $basePath "image_enhancement_server"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$enhancementPath'; python enhancement_server.py 8003"
Start-Sleep -Seconds 3

# 3. 체형 분석 서버 (포트 8002)
Write-Host "[3/3] 체형 분석 서버 시작 (포트 8002)..." -ForegroundColor Yellow
$bodyAnalysisPath = Join-Path $basePath "body_analysis_test"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$bodyAnalysisPath'; python test_body_analysis.py 8002"
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "모든 서버가 시작되었습니다!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "접속 주소:" -ForegroundColor Cyan
Write-Host "  - 메인 백엔드: http://localhost:8000" -ForegroundColor White
Write-Host "  - 이미지 보정: http://localhost:8003" -ForegroundColor White
Write-Host "  - 체형 분석: http://localhost:8002" -ForegroundColor White
Write-Host ""
Write-Host "각 서버 창을 닫으면 해당 서버가 종료됩니다." -ForegroundColor Yellow
Write-Host ""


