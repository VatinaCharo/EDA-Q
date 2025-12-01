@echo off
chcp 65001 >nul
REM EDA-Q Assistant - Version Bump Script (Windows)

echo ====================================
echo   EDA-Q Assistant Version Tool
echo ====================================
echo.

if "%1"=="" (
    echo Usage: bump-version.bat [patch^|minor^|major]
    echo.
    echo Examples:
    echo   bump-version.bat patch   # 0.2.0 -^> 0.2.1
    echo   bump-version.bat minor   # 0.2.0 -^> 0.3.0
    echo   bump-version.bat major   # 0.2.0 -^> 1.0.0
    echo.
    pause
    exit /b 1
)

set VERSION_TYPE=%1

echo Current version:
call npm version --json | findstr "edaq-assistant"

echo.
echo Updating version to %VERSION_TYPE%...
call npm version %VERSION_TYPE% --no-git-tag-version

if %errorlevel% neq 0 (
    echo ERROR: Version update failed
    pause
    exit /b 1
)

echo.
echo Version updated successfully!
echo.
echo New version:
call npm version --json | findstr "edaq-assistant"

echo.
echo Next steps:
echo   1. Check version in package.json
echo   2. Update README.md version history
echo   3. Run package.bat to rebuild
echo.

pause
