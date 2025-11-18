@echo off
cd /d %~dp0
echo ========================================
echo 체형 분석 테스트 서버 시작 (포트 8002)
echo ========================================
python test_body_analysis.py 8002
pause






