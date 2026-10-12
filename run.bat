@echo off
chcp 65001 >nul
title VietNeu Voice Studio (Turbo v3)
echo ========================================================
echo   🎙️  VIETNEU VOICE STUDIO - TURBO V3
echo   🚀  Dung phan cung PC + Cache model D:\Youtube
echo ========================================================
cd /d "%~dp0"

set "HF_HOME=D:\Youtube\libs\models\hf_cache"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "no_proxy="
set "NO_PROXY="

if exist "D:\Youtube\libs\python\python.exe" (
    echo [OK] Phat hien Python va Model Turbo v3 tai D:\Youtube
    "D:\Youtube\libs\python\python.exe" -u app.py
) else (
    echo [!] Khong tim thay D:\Youtube\libs\python, dung python he thong...
    python -u app.py
)

pause
