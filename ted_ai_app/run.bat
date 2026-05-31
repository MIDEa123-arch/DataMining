@echo off
echo ===================================================
echo   TED AI Video App - Khởi động Server
echo ===================================================

echo [1/2] Dang kiem tra va cai dat cac thu vien can thiet...
cd backend
pip install -r requirements.txt

echo.
echo [2/2] Dang khoi dong may chu Web...
echo ===================================================
echo Vui long mo trinh duyet va truy cap: http://localhost:8000
echo Nhan Ctrl+C de tat server khi khong su dung.
echo ===================================================
py main.py
pause
