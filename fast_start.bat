@echo off
chcp 65001 >nul
title VietNeu Voice Studio - Fast Startup
color 0B

echo ===============================================================================
echo                VIETNEU VOICE STUDIO - FAST STARTUP
echo ===============================================================================
echo.

set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

if not exist "D:\Youtube\libs\python\python.exe" (
    color 0C
    echo [LOI] Khong tim thay Python tai: D:\Youtube\libs\python\python.exe
    pause
    exit /b 1
)

taskkill /f /im ngrok.exe >nul 2>&1
taskkill /f /im cloudflared.exe >nul 2>&1

set "HF_HOME=D:\Youtube\libs\models\hf_cache"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "PYTHONIOENCODING=utf-8"
set "no_proxy="
set "NO_PROXY="

echo [1/2] Moi truong: D:\Youtube\libs\python
echo [2/2] Model Cache: VieNeu-TTS-v3-Turbo (Local)
echo.
echo Dang khoi dong may chu va ket noi ten mien Ngrok...
echo.

"D:\Youtube\libs\python\python.exe" -u app.py

pause
