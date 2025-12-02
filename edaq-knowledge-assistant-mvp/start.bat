@echo off
chcp 65001 >nul
echo ================================
echo EDA-Q Knowledge Assistant 一键启动脚本
echo ================================
echo.

echo [1/3] 检查依赖...
if not exist "node_modules" (
    echo 正在安装依赖...
    call npm install
    if errorlevel 1 (
        echo 错误: 依赖安装失败
        pause
        exit /b 1
    )
) else (
    echo ✓ 依赖已安装
)

echo.
echo [2/3] 检查配置...
if not exist ".vscode\settings.json" (
    echo 警告: 未找到API Key配置
    echo 请在VSCode设置中配置 edaq.qwenApiKey
    echo.
)

echo.
echo [3/3] 启动VSCode调试...
echo 按下F5键启动扩展测试
code .

echo.
echo ================================
echo 启动完成\!
echo 请在VSCode中按F5键开始调试
echo ================================
pause
