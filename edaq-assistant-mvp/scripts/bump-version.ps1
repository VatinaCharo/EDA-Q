# EDA-Q Assistant - PowerShell Version Bump Script
# 用于更新版本号

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("patch", "minor", "major")]
    [string]$VersionType
)

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "  EDA-Q Assistant Version Tool" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# 进入项目目录
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectPath = Split-Path -Parent $scriptPath
Set-Location $projectPath

Write-Host "Current version:" -ForegroundColor Yellow
npm version --json | Select-String "edaq-assistant"

Write-Host ""
Write-Host "Updating version to $VersionType..." -ForegroundColor Yellow
npm version $VersionType --no-git-tag-version

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Version update failed" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Version updated successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "New version:" -ForegroundColor Yellow
npm version --json | Select-String "edaq-assistant"

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Check version in package.json"
Write-Host "  2. Update README.md version history"
Write-Host "  3. Run package.ps1 to rebuild"
Write-Host ""

Read-Host "Press Enter to exit"
