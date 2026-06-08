@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo [MyCaptchaOCR] venv not found - create .venv and install deps first. See docs\DESKTOP_UI.md
    pause
    exit /b 1
)
start "MyCaptchaOCR" ".venv\Scripts\pythonw.exe" "scripts\ocr_desktop_app.py"
