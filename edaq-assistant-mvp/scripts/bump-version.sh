#!/bin/bash
# EDA-Q Assistant - 版本号更新脚本 (Linux/Mac)

set -e

echo "===================================="
echo "  EDA-Q Assistant 版本更新工具"
echo "===================================="
echo ""

if [ -z "$1" ]; then
    echo "用法: ./bump-version.sh [patch|minor|major]"
    echo ""
    echo "示例:"
    echo "  ./bump-version.sh patch   # 0.2.0 -> 0.2.1"
    echo "  ./bump-version.sh minor   # 0.2.0 -> 0.3.0"
    echo "  ./bump-version.sh major   # 0.2.0 -> 1.0.0"
    echo ""
    exit 1
fi

VERSION_TYPE=$1

echo "当前版本号:"
npm version --json | grep "edaq-assistant"

echo ""
echo "更新版本号为 $VERSION_TYPE..."
npm version $VERSION_TYPE --no-git-tag-version

echo ""
echo "✅ 版本更新成功!"
echo ""
echo "新版本号:"
npm version --json | grep "edaq-assistant"

echo ""
echo "下一步:"
echo "  1. 检查 package.json 中的版本号"
echo "  2. 更新 README.md 和文档中的版本号"
echo "  3. 运行 ./package.sh 重新打包"
echo ""
