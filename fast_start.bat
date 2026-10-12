@echo off
chcp 65001 >nul
title 🎙️ VietNeu Voice Studio - Fast Startup
color 0B

echo ===============================================================================
echo                🎙️ VIETNEU VOICE STUDIO - FAST STARTUP 🚀
echo ===============================================================================
echo.

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

:: 1. Kiểm tra điều kiện cần 1: Thư mục Python của D:\Youtube
if not exist "D:\Youtube\libs\python\python.exe" (
    color 0C
    echo [LOI] Khong tim thay Python tai: D:\Youtube\libs\python\python.exe
    echo Vui long kiem tra lai o dia D: hoac thu muc D:\Youtube!
    echo.
    pause
    exit /b 1
)

:: 2. Kiểm tra điều kiện cần 2: Model cache của VieNeu Turbo v3
if not exist "D:\Youtube\libs\models\hf_cache" (
    color 0E
    echo [CANH BAO] Khong tim thay cache model tai: D:\Youtube\libs\models\hf_cache
    echo Neu chay, he thong co the se phai tai lai model tu HuggingFace!
    echo.
)

:: 3. Tự động tắt phiên chạy cũ nếu đang chiếm cổng 7860 và dọn tunnel cũ
taskkill /f /im ngrok.exe >nul 2>&1
taskkill /f /im cloudflared.exe >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":7860" ^| findstr "LISTENING"') do (
    if not "%%a"=="" (
        echo [INFO] Phat hien phien chay cu dang mo cong 7860 (PID: %%a). Dang giai phong...
        taskkill /f /pid %%a >nul 2>&1
        timeout /t 1 /nobreak >nul
    )
)

:: 4. Thiết lập biến môi trường tối ưu
set "HF_HOME=D:\Youtube\libs\models\hf_cache"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "PYTHONIOENCODING=utf-8"
set "no_proxy="
set "NO_PROXY="

echo [1/2] Môi trường: D:\Youtube\libs\python
echo [2/2] Model Cache: VieNeu-TTS-v3-Turbo (Local)
echo.
echo ⏳ Đang khởi động máy chủ và tạo link cho iPhone...
echo.

:: 5. Chạy ứng dụng
"D:\Youtube\libs\python\python.exe" -u app.py

echo.
echo ===============================================================================
echo Máy chủ đã dừng. Bấm phím bất kỳ để thoát.
echo ===============================================================================
pause >nul
