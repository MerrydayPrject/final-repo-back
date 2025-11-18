@echo off
cd /d %~dp0
echo ========================================
echo 이미지 보정 서버 시작 (포트 8003)
echo ========================================
python enhancement_server.py 8003
pause






