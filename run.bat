@echo off
chcp 65001 >nul
title VietNeu Voice Studio
echo ========================================================
echo   🎙️ KHỞI ĐỘNG VIETNEU VOICE STUDIO
echo ========================================================
cd /d "%~dp0"
call "venv\Scripts\activate.bat"
python -u app.py
pause
