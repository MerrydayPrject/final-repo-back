@echo off
chcp 65001 > nul
echo ========================================
echo 3D 이미지 변환 테스트 서버 시작
echo 포트: 8003
echo ========================================
echo.

cd /d "%~dp0"
python 3d_conversion.py

pause

