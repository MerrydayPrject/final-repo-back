@echo off
cd /d %~dp0
echo 백엔드 서버 시작 중...
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause





