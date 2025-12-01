#!/bin/bash

echo "================================"
echo "EDA-Q Assistant 一键启动脚本"
echo "================================"
echo ""

echo "[1/3] 检查依赖..."
if [ ! -d "node_modules" ]; then
    echo "正在安装依赖..."
    npm install
    if [ $? -ne 0 ]; then
        echo "错误: 依赖安装失败"
        exit 1
    fi
else
    echo "✓ 依赖已安装"
fi

echo ""
echo "[2/3] 检查配置..."
if [ ! -f ".vscode/settings.json" ]; then
    echo "警告: 未找到API Key配置"
    echo "请在VSCode设置中配置 edaq.qwenApiKey"
    echo ""
fi

echo ""
echo "[3/3] 启动VSCode调试..."
echo "按下F5键启动扩展测试"
code .

echo ""
echo "================================"
echo "启动完成!"
echo "请在VSCode中按F5键开始调试"
echo "================================"
