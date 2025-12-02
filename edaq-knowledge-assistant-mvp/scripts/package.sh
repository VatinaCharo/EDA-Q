#!/bin/bash
# EDA-Q Assistant - Linux/Mac 打包脚本
# 用于快速打包 VSIX 文件

set -e

echo "===================================="
echo "  EDA-Q Assistant 打包工具"
echo "===================================="
echo ""

# 检查 Node.js
echo "[1/5] 检查 Node.js..."
if ! command -v node &> /dev/null; then
    echo "❌ 错误: 未找到 Node.js"
    echo "请先安装 Node.js: https://nodejs.org/"
    exit 1
fi
echo "✓ Node.js 已安装: $(node --version)"

# 检查 npm
echo ""
echo "[2/5] 检查 npm..."
if ! command -v npm &> /dev/null; then
    echo "❌ 错误: 未找到 npm"
    exit 1
fi
echo "✓ npm 已安装: $(npm --version)"

# 检查并安装依赖
echo ""
echo "[3/5] 检查依赖..."
if [ ! -d "node_modules" ]; then
    echo "📦 安装依赖..."
    npm install
else
    echo "✓ 依赖已安装"
fi

# 检查并安装 vsce
echo ""
echo "[4/5] 检查 vsce..."
if ! command -v vsce &> /dev/null; then
    echo "📦 安装 vsce..."
    npm install -g @vscode/vsce
else
    echo "✓ vsce 已安装: $(vsce --version)"
fi

# 打包
echo ""
echo "[5/5] 开始打包..."
vsce package

echo ""
echo "===================================="
echo "  ✅ 打包成功!"
echo "===================================="
echo ""
echo "📦 VSIX 文件已生成"
echo ""
echo "安装方法:"
echo "  1. 打开 VSCode"
echo "  2. 按 Ctrl+Shift+P (Mac: Cmd+Shift+P)"
echo "  3. 输入 'Extensions: Install from VSIX...'"
echo "  4. 选择生成的 .vsix 文件"
echo ""

# 列出生成的文件
for f in *.vsix; do
    if [ -f "$f" ]; then
        echo "生成的文件: $f"
        echo "文件大小: $(du -h "$f" | cut -f1)"
    fi
done

echo ""
