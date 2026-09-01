@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo   SilentCutter - установка
echo ============================================
echo.

where winget >nul 2>nul
if %errorlevel% neq 0 (
    echo На этом компьютере нет winget ^(обычно он есть на Windows 10/11^).
    echo Установите вручную:
    echo   1. Python: https://www.python.org/downloads/  ^(отметьте "Add to PATH"^)
    echo   2. ffmpeg: https://www.gyan.dev/ffmpeg/builds/  ^(essentials build,
    echo      распакуйте и добавьте папку bin в PATH^)
    echo После этого запустите install.bat ещё раз.
    pause
    exit /b 1
)

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python не найден. Устанавливаю через winget...
    echo ^(это может занять пару минут, дождитесь окончания^)
    winget install -e --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
    echo.
    echo Python установлен, но Windows ещё не обновил PATH в этом окне.
    echo Закройте это окно и запустите install.bat ЕЩЁ РАЗ.
    pause
    exit /b 0
)

where ffmpeg >nul 2>nul
if %errorlevel% neq 0 (
    echo ffmpeg не найден. Устанавливаю через winget...
    echo ^(это может занять пару минут, дождитесь окончания^)
    winget install -e --id Gyan.FFmpeg --silent --accept-package-agreements --accept-source-agreements
    echo.
    echo ffmpeg установлен, но Windows ещё не обновил PATH в этом окне.
    echo Закройте это окно и запустите install.bat ЕЩЁ РАЗ.
    pause
    exit /b 0
)

echo Python и ffmpeg на месте.
echo.
echo Создаю виртуальное окружение и ставлю зависимости...
python -m venv .venv
if %errorlevel% neq 0 (
    echo.
    echo ОШИБКА: не получилось создать виртуальное окружение.
    echo Попробуйте перезапустить компьютер и запустить install.bat заново.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\pip.exe" install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo ОШИБКА при установке зависимостей. Проверьте интернет-соединение
    echo и запустите install.bat заново.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Готово! Установка завершена успешно.
echo   Теперь запускайте программу двойным кликом
echo   по файлу Run_SilentCutter.bat
echo ============================================
pause
