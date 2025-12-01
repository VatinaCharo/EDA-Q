@echo off
chcp 65001 >nul
REM EDA-Q Assistant - Windows Package Script

echo ====================================
echo   EDA-Q Assistant Package Tool
echo ====================================
echo.

REM Check Node.js
echo [1/5] Checking Node.js...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js not found
    echo Please install Node.js: https://nodejs.org/
    pause
    exit /b 1
)
echo OK - Node.js installed

REM Check npm
echo.
echo [2/5] Checking npm...
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: npm not found
    pause
    exit /b 1
)
echo OK - npm installed

REM Check dependencies
echo.
echo [3/5] Checking dependencies...
if not exist node_modules (
    echo Installing dependencies...
    call npm install
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo OK - Dependencies installed
)

REM Check vsce
echo.
echo [4/5] Checking vsce...
call vsce --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing vsce...
    call npm install -g @vscode/vsce
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install vsce
        pause
        exit /b 1
    )
) else (
    echo OK - vsce installed
)

REM Package
echo.
echo [5/5] Packaging...
call vsce package
if %errorlevel% neq 0 (
    echo ERROR: Package failed
    pause
    exit /b 1
)

echo.
echo ====================================
echo   Package Success!
echo ====================================
echo.
echo VSIX file generated
echo.
echo Installation:
echo   1. Open VSCode
echo   2. Press Ctrl+Shift+P
echo   3. Type: Extensions: Install from VSIX...
echo   4. Select the .vsix file
echo.

REM List generated files
for %%f in (*.vsix) do (
    echo Generated: %%f
    echo Size: %%~zf bytes
)

echo.
pause
