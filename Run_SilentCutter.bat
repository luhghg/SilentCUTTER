@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Программа ещё не установлена.
    echo Сначала запустите install.bat ^(один раз^), потом уже этот файл.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m silence_cutter.main
if %errorlevel% neq 0 (
    echo.
    echo Программа завершилась с ошибкой ^(см. текст выше^).
    pause
)
