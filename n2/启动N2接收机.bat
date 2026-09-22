@echo off
cd /d "%~dp0"
where pythonw >nul 2>&1
if errorlevel 1 goto console
start "" pythonw "%~dp0app.py"
exit /b 0

:console
echo pythonw was not found. Starting with python instead.
python "%~dp0app.py"
if errorlevel 1 pause
