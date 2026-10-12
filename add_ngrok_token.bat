@echo off
chcp 65001 >nul
title 🔑 Cấu hình Ngrok Authtoken
color 0A

echo ===============================================================================
echo                🔑 CÀI ĐẶT AUTHTOKEN NGROK CHO TÊN MIỀN CỐ ĐỊNH 🚀
echo ===============================================================================
echo.
echo Bước 1: Mở trình duyệt và lấy Authtoken tại:
echo         https://dashboard.ngrok.com/get-started/your-authtoken
echo.
echo Bước 2: Dán token của bạn vào bên dưới rồi bấm Enter.
echo.
set /p TOKEN="👉 Nhập hoặc dán Authtoken: "

if "%TOKEN%"=="" (
    echo.
    echo [LỖI] Bạn chưa nhập token! Vui lòng mở lại file để thử lại.
    echo.
    pause
    exit /b 1
)

echo.
echo ⏳ Đang lưu token vào hệ thống...
"%~dp0tools\ngrok.exe" config add-authtoken %TOKEN%

if %errorlevel% equ 0 (
    echo.
    echo ===============================================================================
    echo ✅ ĐÃ CẤU HÌNH THÀNH CÔNG AUTHTOKEN NGROK!
    echo Bây giờ bạn có thể chạy 'fast_start.bat' hoặc bật ứng dụng từ Desktop.
    echo Tên miền cố định: https://nondisciplining-foudroyant-calvin.ngrok-free.dev
    echo ===============================================================================
) else (
    echo.
    echo [LỖI] Không thể lưu authtoken. Vui lòng kiểm tra lại token vừa nhập!
)

echo.
pause
