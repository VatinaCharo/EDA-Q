# EDA-Q Assistant - PowerShell Package Script
# 用于在 Windows PowerShell 中打包扩展

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "  EDA-Q Assistant Package Tool" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Node.js
Write-Host "[1/5] Checking Node.js..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "OK - Node.js installed: $nodeVersion" -ForegroundColor Green
    } else {
        throw "Node.js not found"
    }
} catch {
    Write-Host "ERROR: Node.js not found" -ForegroundColor Red
    Write-Host "Please install Node.js: https://nodejs.org/" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# 检查 npm
Write-Host ""
Write-Host "[2/5] Checking npm..." -ForegroundColor Yellow
try {
    $npmVersion = npm --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "OK - npm installed: $npmVersion" -ForegroundColor Green
    } else {
        throw "npm not found"
    }
} catch {
    Write-Host "ERROR: npm not found" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# 进入项目目录
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectPath = Split-Path -Parent $scriptPath
Set-Location $projectPath

# 检查依赖
Write-Host ""
Write-Host "[3/5] Checking dependencies..." -ForegroundColor Yellow
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Host "OK - Dependencies installed" -ForegroundColor Green
}

# 检查 vsce
Write-Host ""
Write-Host "[4/5] Checking vsce..." -ForegroundColor Yellow
try {
    $vsceVersion = vsce --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "OK - vsce installed: $vsceVersion" -ForegroundColor Green
    } else {
        throw "vsce not found"
    }
} catch {
    Write-Host "Installing vsce..." -ForegroundColor Yellow
    npm install -g @vscode/vsce
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install vsce" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# 打包
Write-Host ""
Write-Host "[5/5] Packaging..." -ForegroundColor Yellow
vsce package
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Package failed" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "====================================" -ForegroundColor Green
Write-Host "  Package Success!" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host ""
Write-Host "VSIX file generated" -ForegroundColor Cyan
Write-Host ""
Write-Host "Installation:" -ForegroundColor Yellow
Write-Host "  1. Open VSCode"
Write-Host "  2. Press Ctrl+Shift+P"
Write-Host "  3. Type: Extensions: Install from VSIX..."
Write-Host "  4. Select the .vsix file"
Write-Host ""

# 列出生成的文件
$vsixFiles = Get-ChildItem -Filter "*.vsix"
foreach ($file in $vsixFiles) {
    Write-Host "Generated: $($file.Name)" -ForegroundColor Cyan
    Write-Host "Size: $([math]::Round($file.Length/1KB, 2)) KB" -ForegroundColor Cyan
}

Write-Host ""
Read-Host "Press Enter to exit"
