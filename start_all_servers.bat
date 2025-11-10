@echo off
echo ========================================
echo 모든 서버 시작 중...
echo ========================================
echo.

cd /d %~dp0

echo [1/3] 메인 백엔드 서버 시작 (포트 8000)...
start "메인 백엔드 서버 (8000)" cmd /k "cd /d %~dp0 && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 3 /nobreak >nul

echo [2/3] 이미지 보정 서버 시작 (포트 8003)...
start "이미지 보정 서버 (8003)" cmd /k "cd /d %~dp0\image_enhancement_server && python enhancement_server.py 8003"
timeout /t 3 /nobreak >nul

echo [3/3] 체형 분석 서버 시작 (포트 8002)...
start "체형 분석 서버 (8002)" cmd /k "cd /d %~dp0\body_analysis_test && python test_body_analysis.py 8002"
timeout /t 2 /nobreak >nul

echo.
echo ========================================
echo 모든 서버가 시작되었습니다!
echo ========================================
echo.
echo 접속 주소:
echo   - 메인 백엔드: http://localhost:8000
echo   - 이미지 보정: http://localhost:8003
echo   - 체형 분석: http://localhost:8002
echo.
echo 각 서버 창을 닫으면 해당 서버가 종료됩니다.
echo.
pause


