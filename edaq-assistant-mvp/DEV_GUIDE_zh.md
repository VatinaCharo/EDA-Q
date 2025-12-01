# EDA-Q Assistant 开发文档

> VSCode 扩展开发完整指南 - 从架构到发布

---

## 📖 目录

- [项目概述](#项目概述)
- [技术架构](#技术架构)
- [目录结构](#目录结构)
- [核心文件详解](#核心文件详解)
- [开发环境搭建](#开发环境搭建)
- [开发工作流](#开发工作流)
- [多语言实现](#多语言实现)
- [打包发布](#打包发布)
- [常见开发问题](#常见开发问题)
- [扩展开发指南](#扩展开发指南)

---

## 📋 项目概述

### 项目信息

- **名称**: EDA-Q Assistant
- **类型**: VSCode Extension (扩展插件)
- **版本**: 0.2.0
- **技术栈**: Node.js + VSCode Extension API
- **AI 模型**: 阿里云千问 (可扩展其他 LLM)
- **代码量**: ~1000 行核心代码

### 功能特性

✅ **核心功能**
- AI 对话式代码生成
- 上下文感知 (自动读取编辑器代码)
- 代码一键插入编辑器
- 流式输出模拟
- 对话历史管理

✅ **多语言支持**
- 中英文界面切换
- 双语知识库 (515行中文 + 515行英文 API 文档)
- 动态语言包加载
- AI 回复语言自适应

✅ **开发友好**
- 热重载调试
- 完整的打包脚本
- 版本管理自动化
- 跨平台支持 (Windows/Mac/Linux)

---

## 🏗️ 技术架构

### 整体架构图

```
┌─────────────────────────────────────────────────────┐
│                   VSCode Extension                   │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────────┐      ┌──────────────┐             │
│  │ extension.js│──────│ qwenClient.js│             │
│  │  (主入口)   │      │  (AI客户端)  │             │
│  └─────────────┘      └──────────────┘             │
│         │                      │                     │
│         │                      │                     │
│  ┌──────▼──────┐      ┌───────▼────────┐           │
│  │  Webview UI │      │  Qwen API      │           │
│  │  (聊天界面) │      │  (阿里云千问)  │           │
│  └─────────────┘      └────────────────┘           │
│         │                                            │
│  ┌──────▼──────────────────────────┐               │
│  │  Language Packs (locales/)      │               │
│  │  Knowledge Base (knowledge/)    │               │
│  └─────────────────────────────────┘               │
└─────────────────────────────────────────────────────┘
```

### 技术选型

| 组件 | 技术 | 说明 |
|------|------|------|
| 扩展框架 | VSCode Extension API | 官方扩展 API |
| 编程语言 | JavaScript (Node.js) | 轻量级,易维护 |
| UI 框架 | Webview (HTML/CSS/JS) | 原生 VSCode Webview |
| AI 服务 | 阿里云千问 API | HTTP REST API |
| HTTP 客户端 | Axios | Promise-based HTTP 库 |
| 国际化 | 自定义 i18n 模块 | 轻量级,无依赖 |
| 打包工具 | @vscode/vsce | 官方打包工具 |

### 数据流

```
用户输入
   ↓
Webview UI (extension.js)
   ↓
提取上下文 (当前编辑器代码)
   ↓
构建提示词 (系统提示 + 知识库 + 用户问题)
   ↓
QwenClient (qwenClient.js)
   ↓
调用千问 API
   ↓
接收 AI 回复
   ↓
解析代码块
   ↓
Webview 显示 (流式模拟)
   ↓
用户操作 (复制/插入代码)
```

---

## 📁 目录结构

### 完整目录树

```
edaq-assistant-mvp/
├── extension.js              # 🔥 扩展主入口 (810 行)
├── qwenClient.js             # 🔥 AI 客户端 (463 行)
├── package.json              # 🔥 扩展配置文件
├── LICENSE                   # MIT 许可证
├── .vscodeignore            # 打包排除文件配置
├── icon.png                 # 扩展图标 (128x128)
│
├── locales/                 # 🌐 语言包目录
│   ├── zh-CN.js            # 中文语言包 (UI 文本)
│   └── en-US.js            # 英文语言包 (UI 文本)
│
├── knowledge/              # 📚 AI 知识库
│   ├── user_manual_zh.txt  # 中文 API 文档 (515 行)
│   └── user_manual_en.txt  # 英文 API 文档 (515 行)
│
├── scripts/                # 🛠️ 自动化脚本
│   ├── package.ps1         # PowerShell 打包脚本
│   ├── package.bat         # Windows 批处理打包
│   ├── package.sh          # Linux/Mac 打包脚本
│   ├── bump-version.ps1    # PowerShell 版本更新
│   ├── bump-version.bat    # Windows 版本更新
│   └── bump-version.sh     # Linux/Mac 版本更新
│
├── node_modules/           # 📦 依赖包 (npm install)
│
├── .vscode/                # VSCode 配置
│   └── extensions.json     # 推荐扩展
│
├── README.md              # 📄 项目说明 (开发者版)
├── README_zh.md           # 📄 用户指南 (中文)
├── README_en.md           # 📄 用户指南 (英文)
│
├── demo_examples.py       # 🧪 示例代码
├── start.bat             # 快速启动脚本 (Windows)
├── start.sh              # 快速启动脚本 (Linux/Mac)
│
└── edaq-assistant-0.2.0.vsix  # 📦 打包后的安装文件
```

### 核心文件说明

| 文件 | 大小 | 作用 | 修改频率 |
|------|------|------|----------|
| `extension.js` | ~810 行 | 扩展主逻辑和 UI | 高 |
| `qwenClient.js` | ~463 行 | AI 客户端和知识库 | 中 |
| `package.json` | ~120 行 | 配置和元数据 | 中 |
| `locales/*.js` | ~150 行/文件 | UI 文本翻译 | 低 |
| `knowledge/*.txt` | ~515 行/文件 | API 文档 | 低 |

---

## 🔥 核心文件详解

### 1. extension.js - 扩展主入口

**职责**: 扩展生命周期管理、UI 渲染、用户交互

#### 关键组成部分

```javascript
// 1. 激活函数 - 扩展入口
function activate(context) {
    // 加载语言包
    // 注册命令
    // 创建 WebviewProvider
    // 监听配置变化
}

// 2. ChatViewProvider - UI 管理类
class ChatViewProvider {
    constructor(context) {
        // 初始化语言、对话历史
        // 监听语言切换
    }

    // Webview 解析和渲染
    resolveWebviewView(webviewView) {
        // 设置 HTML 内容
        // 监听消息
    }

    // 处理用户消息
    async _processUserMessage(userMessage) {
        // 获取配置
        // 获取上下文
        // 调用 AI
        // 返回结果
    }

    // 生成 HTML 内容
    _getHtmlContent(webview) {
        // 使用语言包渲染 UI
        // 返回完整 HTML
    }
}
```

#### 核心流程

```javascript
// 用户发送消息流程
用户输入 → sendMessage()
  ↓
显示 thinking 状态
  ↓
_processUserMessage()
  ↓
创建 QwenClient(apiKey, model, path, language)
  ↓
调用 client.chat(message, context, history)
  ↓
接收响应
  ↓
解析代码块
  ↓
流式显示 (模拟)
  ↓
添加到对话历史
```

#### 重要方法

| 方法 | 参数 | 返回值 | 作用 |
|------|------|--------|------|
| `activate()` | context | void | 扩展激活入口 |
| `loadLanguagePack()` | context, language | i18n | 加载语言包 |
| `_processUserMessage()` | userMessage | Promise | 处理用户消息 |
| `_getContext()` | - | {code, fileName} | 获取编辑器上下文 |
| `_insertCodeToEditor()` | code | void | 插入代码到编辑器 |
| `_getHtmlContent()` | webview | string | 生成 UI HTML |

### 2. qwenClient.js - AI 客户端

**职责**: AI API 调用、知识库管理、提示词构建

#### 类结构

```javascript
class QwenClient {
    constructor(apiKey, model, extensionPath, language) {
        this.apiKey = apiKey;
        this.model = model;  // qwen-plus/turbo/max
        this.language = language;  // zh-CN/en-US
        this.i18n = this._loadLanguagePack();
        this.knowledgeBase = this._loadKnowledgeBase();
    }

    // 加载语言包
    _loadLanguagePack() { }

    // 加载知识库
    _loadKnowledgeBase() { }

    // 构建系统提示词
    _buildSystemPrompt() { }

    // 主聊天方法
    async chat(userMessage, context, history) { }

    // 错误处理
    _handleError(error) { }
}
```

#### 系统提示词结构

```javascript
_buildSystemPrompt() {
    return `
    你是 EDA-Q 量子芯片设计工具的专业AI助手。

    ## 核心知识库
    ${this.knowledgeBase}  // 加载对应语言的 API 文档

    ## 重要规则
    - 严格遵循知识库示例
    - 参数名称、类型、顺序必须正确
    - 代码用 \`\`\`python 包裹
    - 使用${language}注释

    ## 典型工作流程
    [包含完整的代码模板和顺序规则]
    `;
}
```

#### API 调用流程

```javascript
// HTTP 请求结构
POST https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation

Headers:
  Authorization: Bearer ${apiKey}
  Content-Type: application/json

Body:
{
  model: "qwen-plus",
  input: {
    messages: [
      {role: "system", content: systemPrompt},
      ...conversationHistory,
      {role: "user", content: userMessage}
    ]
  },
  parameters: {
    temperature: 0.7,
    top_p: 0.8,
    max_tokens: 2000
  }
}
```

### 3. package.json - 扩展配置

**关键配置项**:

```json
{
  "name": "edaq-assistant",           // 扩展 ID
  "displayName": "EDA-Q Assistant",   // 显示名称
  "version": "0.2.0",                 // 版本号
  "publisher": "edaq-team",           // 发布者

  "engines": {
    "vscode": "^1.80.0"              // 最低 VSCode 版本
  },

  "main": "./extension.js",           // 入口文件

  "contributes": {
    "commands": [ ],                  // 注册命令
    "viewsContainers": { },          // 侧边栏容器
    "views": { },                     // 视图
    "configuration": { }              // 配置项
  },

  "scripts": {
    "package": "vsce package",        // 打包命令
    "publish": "vsce publish"         // 发布命令
  },

  "dependencies": {
    "axios": "^1.13.2"               // HTTP 库
  }
}
```

### 4. 语言包结构 (locales/)

**zh-CN.js / en-US.js**:

```javascript
module.exports = {
    ui: {
        header: { title, subtitle },
        welcome: { icon, title, description, examples },
        quickActions: [ {icon, text, question/action} ],
        input: { placeholder, sendButton, sendingButton },
        codeBlock: { copyButton, insertButton, copied },
        thinking: { analyzing, connecting, generating },
        messages: { errorPrefix, ... }
    },
    errors: {
        noApiKey, apiKeyInvalid, rateLimitExceeded, ...
    },
    config: {
        apiKey: { description, ... },
        model: { description, options },
        language: { description, options },
        enableContext: { description }
    }
};
```

### 5. 知识库结构 (knowledge/)

**user_manual_zh.txt / user_manual_en.txt**:

```
# EDA-Q API 快速参考

## 1. Design 类 - 主设计对象
### 初始化
### 核心方法
  - generate_topology() - 生成拓扑
  - generate_qubits() - 生成量子比特
  - generate_coupling_lines() - 生成耦合器
  - ... (共 515 行详细 API 说明)

## 2. Topology 类
## 3. GDS 类
## 4. 完整流程示例
## 5. 常见问题和解决方案
## 6. 设计最佳实践
```

---

## 🛠️ 开发环境搭建

### 前置要求

- **Node.js**: ≥ 16.0.0
- **npm**: ≥ 8.0.0
- **VSCode**: ≥ 1.80.0
- **Git**: 用于版本控制

### 快速开始

```bash
# 1. 克隆/下载项目
cd edaq-assistant-mvp

# 2. 安装依赖
npm install

# 3. 打开 VSCode
code .

# 4. 按 F5 启动调试
# VSCode 会打开新窗口,扩展在新窗口中加载
```

### 依赖说明

```json
{
  "dependencies": {
    "axios": "^1.13.2"          // HTTP 请求库
  },
  "devDependencies": {
    "@types/vscode": "^1.80.0", // VSCode API 类型定义
    "eslint": "^8.50.0"         // 代码检查
  }
}
```

### 调试配置

VSCode 自动生成 `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Run Extension",
      "type": "extensionHost",
      "request": "launch",
      "args": ["--extensionDevelopmentPath=${workspaceFolder}"]
    }
  ]
}
```

---

## 🔄 开发工作流

### 日常开发流程

```
修改代码
   ↓
按 F5 / Ctrl+R 重新加载
   ↓
在扩展开发主机中测试
   ↓
查看控制台输出 (调试日志)
   ↓
查看 Webview 开发者工具 (UI 调试)
   ↓
修复问题
   ↓
重复
```

### 调试技巧

#### 1. 扩展主机调试 (extension.js)

```javascript
// 在代码中添加断点或 console.log
console.log('✅ 扩展已激活');
console.log('📝 用户消息:', userMessage);
console.log('🔑 API Key:', apiKey ? '已配置' : '未配置');
```

查看输出: **调试控制台** (Debug Console)

#### 2. Webview UI 调试

```javascript
// 在 Webview 的 <script> 中
console.log('💬 发送消息:', message);
console.log('📦 收到响应:', data);
```

查看输出:
1. 在扩展开发主机窗口
2. 按 `Ctrl+Shift+P`
3. 输入 "Developer: Open Webview Developer Tools"

#### 3. 网络请求调试

```javascript
// qwenClient.js
console.log('🚀 发送请求到千问API...');
console.log('📝 消息数量:', messages.length);
console.log('✅ 收到API响应');
console.log('📊 Token使用:', usage);
```

### 代码风格

```javascript
// 1. 使用清晰的注释
// 用户输入验证
if (!message.trim()) return;

// 2. 使用 emoji 标记日志
console.log('✅ 成功');
console.log('❌ 错误');
console.log('📝 信息');

// 3. 异步函数使用 async/await
async function processMessage() {
    try {
        const result = await client.chat();
    } catch (error) {
        console.error('❌ 错误:', error);
    }
}

// 4. 错误处理要完善
if (!apiKey) {
    throw new Error(i18n.errors.noApiKey);
}
```

---

## 🌐 多语言实现

### 架构设计

```
配置 (edaq.language: zh-CN/en-US)
    ↓
extension.js 加载语言包
    ↓
QwenClient 使用相同语言
    ↓
Webview UI 使用语言包渲染
    ↓
知识库加载对应语言文档
    ↓
AI 使用对应语言回复
```

### 实现细节

#### 1. 语言包加载 (extension.js)

```javascript
function loadLanguagePack(context, language = 'zh-CN') {
    const langFile = language === 'en-US' ? 'en-US.js' : 'zh-CN.js';
    const langPath = path.join(context.extensionPath, 'locales', langFile);

    // 清除缓存
    delete require.cache[require.resolve(langPath)];
    return require(langPath);
}

// 监听语言变化
vscode.workspace.onDidChangeConfiguration(e => {
    if (e.affectsConfiguration('edaq.language')) {
        this._updateLanguage();
        // 重新渲染 UI
        this._view.webview.html = this._getHtmlContent();
    }
});
```

#### 2. 知识库加载 (qwenClient.js)

```javascript
_loadKnowledgeBase() {
    const manualFile = this.language === 'en-US'
        ? 'user_manual_en.txt'
        : 'user_manual_zh.txt';

    const manualPath = path.join(
        this.extensionPath,
        'knowledge',
        manualFile
    );

    return fs.readFileSync(manualPath, 'utf-8');
}
```

#### 3. UI 国际化 (extension.js)

```javascript
_getHtmlContent(webview) {
    const ui = this._i18n.ui;

    return `
        <h2>${ui.header.title}</h2>
        <p>${ui.header.subtitle}</p>

        <script>
            const i18n = ${JSON.stringify(ui)};

            // JavaScript 中使用
            sendButton.textContent = i18n.input.sendButton;
        </script>
    `;
}
```

### 添加新语言步骤

```bash
# 1. 创建语言包
cp locales/zh-CN.js locales/ja-JP.js

# 2. 翻译所有文本
# 编辑 locales/ja-JP.js

# 3. 创建知识库
cp knowledge/user_manual_zh.txt knowledge/user_manual_ja.txt

# 4. 翻译 API 文档
# 编辑 knowledge/user_manual_ja.txt

# 5. 更新 package.json
# 添加 "ja-JP" 到 enum 和 enumDescriptions

# 6. 更新加载逻辑
# extension.js 和 qwenClient.js 添加日语支持
```

---

## 📦 打包发布

### 打包流程

#### 方法 1: 使用自动化脚本 (推荐)

**Windows PowerShell:**
```powershell
.\scripts\package.ps1
```

**Linux/Mac:**
```bash
./scripts/package.sh
```

**功能**:
1. ✅ 检查 Node.js 和 npm
2. ✅ 自动安装依赖
3. ✅ 自动安装 vsce
4. ✅ 执行打包
5. ✅ 显示生成的文件信息

#### 方法 2: 手动打包

```bash
# 1. 安装 vsce
npm install -g @vscode/vsce

# 2. 安装依赖
npm install

# 3. 打包
vsce package

# 生成: edaq-assistant-0.2.0.vsix
```

### 版本管理

#### 自动更新版本号

```bash
# 补丁版本 (0.2.0 -> 0.2.1) - Bug 修复
.\scripts\bump-version.ps1 patch

# 次版本 (0.2.0 -> 0.3.0) - 新功能
.\scripts\bump-version.ps1 minor

# 主版本 (0.2.0 -> 1.0.0) - 重大更新
.\scripts\bump-version.ps1 major
```

#### 手动更新

编辑 `package.json`:
```json
{
  "version": "0.2.1"
}
```

### 发布流程

```bash
# 1. 更新版本号
npm version patch  # 或 minor/major

# 2. 更新文档
# - README.md 版本历史
# - README_zh.md 版本号
# - README_en.md 版本号

# 3. 重新打包
npm run package

# 4. 测试
# 安装生成的 VSIX 并完整测试

# 5. 分发
# - 邮件发送
# - 内部服务器
# - GitHub Releases
```

### .vscodeignore 配置

排除不必要的文件以减小包体积:

```
# 开发文件
.vscode/**
.vscode-test/**
node_modules/**/.bin/**

# 文档 (保留 README.md)
*.md
!README.md

# 示例和测试
demo_examples.py
*.vsix

# Git 文件
.gitignore
.editorconfig
```

### 打包优化

| 优化项 | 方法 | 效果 |
|--------|------|------|
| 排除开发文件 | .vscodeignore | 减小 50% |
| 移除 devDependencies | npm prune --production | 减小 30% |
| 压缩图片 | icon.png 优化 | 减小 10KB |

---

## ❓ 常见开发问题

### 1. 扩展无法激活

**症状**: 按 F5 后扩展不显示

**排查步骤**:
```javascript
// 1. 检查 package.json 中的 activationEvents
"activationEvents": ["onStartupFinished"]

// 2. 检查 activate() 函数
function activate(context) {
    console.log('✅ 激活成功');
}

// 3. 查看调试控制台错误
```

### 2. Webview 显示空白

**原因**: CSP (内容安全策略) 或脚本错误

**解决**:
```html
<!-- 1. 检查 CSP 设置 -->
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none';
               style-src 'unsafe-inline';
               script-src 'unsafe-inline' 'unsafe-eval';">

<!-- 2. 检查 JavaScript 错误 -->
<!-- 打开 Webview 开发者工具查看 -->
```

### 3. API 调用失败

**常见错误**:

```javascript
// 错误 1: API Key 未配置
if (!apiKey || apiKey.trim() === '') {
    throw new Error(this._i18n.errors.noApiKey);
}

// 错误 2: 网络问题
try {
    const response = await axios.post(url, data);
} catch (error) {
    if (error.code === 'ENOTFOUND') {
        // 网络不可达
    }
}

// 错误 3: 401 Unauthorized
if (status === 401) {
    // API Key 无效
}
```

### 4. 语言切换不生效

**原因**: 缓存未清除

**解决**:
```javascript
// 加载语言包时清除 require 缓存
delete require.cache[require.resolve(langPath)];
return require(langPath);
```

### 5. 代码插入失败

**原因**: 未打开 Python 文件

**解决**:
```javascript
const editor = vscode.window.activeTextEditor;
if (!editor) {
    vscode.window.showWarningMessage(
        this._i18n.ui.messages.openFileWarning
    );
    return;
}

// 可选: 检查文件类型
if (editor.document.languageId !== 'python') {
    vscode.window.showWarningMessage('请打开 Python 文件');
    return;
}
```

### 6. 打包错误

**错误: Missing publisher**
```json
// package.json 中必须有 publisher
{
  "publisher": "edaq-team"
}
```

**错误: Icon not found**
```json
// 确保 icon.png 存在,或删除配置
{
  // "icon": "icon.png"  // 注释掉
}
```

---

## 🚀 扩展开发指南

### 添加新功能

#### 示例: 添加 "代码解释" 功能

**1. 在 UI 中添加按钮**:

```javascript
// extension.js - _getHtmlContent()
quickActions: [
    { icon: "📖", text: "解释代码", question: "解释这段代码的作用" }
]
```

**2. 添加处理逻辑**:

```javascript
// 已有的 _processUserMessage() 会自动处理
// 只需确保系统提示词包含相关指令
```

**3. 更新语言包**:

```javascript
// locales/zh-CN.js
quickActions: [
    // ...
    { icon: "📖", text: "解释代码", question: "解释这段代码的作用" }
]

// locales/en-US.js
quickActions: [
    // ...
    { icon: "📖", text: "Explain Code", question: "Explain what this code does" }
]
```

#### 示例: 添加配置项

**1. 在 package.json 中添加**:

```json
{
  "configuration": {
    "properties": {
      "edaq.maxTokens": {
        "type": "number",
        "default": 2000,
        "description": "最大 Token 数量"
      }
    }
  }
}
```

**2. 在代码中读取**:

```javascript
// extension.js
const config = vscode.workspace.getConfiguration('edaq');
const maxTokens = config.get('maxTokens', 2000);

// 传递给 QwenClient
const client = new QwenClient(apiKey, model, path, language, maxTokens);
```

### 更换 AI 模型

#### 从千问切换到其他 LLM

**1. 创建新的 AI 客户端**:

```javascript
// openaiClient.js
class OpenAIClient {
    constructor(apiKey, model, extensionPath, language) {
        this.apiKey = apiKey;
        this.baseURL = 'https://api.openai.com/v1/chat/completions';
        // ...
    }

    async chat(userMessage, context, history) {
        // 实现 OpenAI API 调用
    }
}
```

**2. 更新配置**:

```json
// package.json
{
  "edaq.aiProvider": {
    "type": "string",
    "enum": ["qwen", "openai", "claude"],
    "default": "qwen"
  }
}
```

**3. 动态选择客户端**:

```javascript
// extension.js
const provider = config.get('aiProvider');
let client;

switch(provider) {
    case 'openai':
        client = new OpenAIClient(...);
        break;
    case 'qwen':
    default:
        client = new QwenClient(...);
}
```

### 性能优化

#### 1. 减小包体积

```bash
# 只安装生产依赖
npm install --production

# 检查包大小
vsce package --out test.vsix
du -h test.vsix
```

#### 2. 异步加载

```javascript
// 延迟加载大型依赖
let heavyModule;

async function useHeavyFeature() {
    if (!heavyModule) {
        heavyModule = await import('./heavy-module');
    }
    return heavyModule.doSomething();
}
```

#### 3. 缓存优化

```javascript
// 缓存知识库内容
class QwenClient {
    static knowledgeCache = new Map();

    _loadKnowledgeBase() {
        const key = this.language;
        if (QwenClient.knowledgeCache.has(key)) {
            return QwenClient.knowledgeCache.get(key);
        }

        const content = fs.readFileSync(...);
        QwenClient.knowledgeCache.set(key, content);
        return content;
    }
}
```

---

## 📊 代码统计

### 核心代码量

| 文件 | 行数 | 说明 |
|------|------|------|
| extension.js | 810 | 主逻辑和 UI |
| qwenClient.js | 463 | AI 客户端 |
| locales/zh-CN.js | 150 | 中文语言包 |
| locales/en-US.js | 150 | 英文语言包 |
| knowledge/user_manual_zh.txt | 515 | 中文文档 |
| knowledge/user_manual_en.txt | 515 | 英文文档 |
| **总计** | **~2600** | **核心代码** |

### 文件类型分布

- **JavaScript**: 1423 行 (55%)
- **文本文档**: 1030 行 (40%)
- **JSON 配置**: 120 行 (5%)

---

## 🎓 最佳实践

### 1. 代码组织

✅ **推荐**:
```javascript
// 单一职责
class QwenClient {
    // 只负责 AI 调用
}

class ChatViewProvider {
    // 只负责 UI 管理
}
```

❌ **避免**:
```javascript
// 所有逻辑放在 activate() 里
function activate(context) {
    // 1000+ 行代码...
}
```

### 2. 错误处理

✅ **推荐**:
```javascript
try {
    const result = await apiCall();
    return result;
} catch (error) {
    console.error('❌ API 错误:', error);
    throw new Error(this.i18n.errors.networkError);
}
```

❌ **避免**:
```javascript
// 忽略错误
apiCall().catch(() => {});
```

### 3. 用户体验

✅ **推荐**:
```javascript
// 提供清晰的加载状态
this._view.webview.postMessage({ type: 'thinking' });

// 提供错误提示
vscode.window.showErrorMessage(i18n.errors.apiKeyInvalid);
```

❌ **避免**:
```javascript
// 无任何反馈
await longRunningTask();
```

---

## 📚 参考资源

### 官方文档

- [VSCode Extension API](https://code.visualstudio.com/api)
- [Webview API](https://code.visualstudio.com/api/extension-guides/webview)
- [Publishing Extensions](https://code.visualstudio.com/api/working-with-extensions/publishing-extension)

### 工具

- [@vscode/vsce](https://github.com/microsoft/vscode-vsce) - 打包工具
- [Axios](https://axios-http.com/) - HTTP 库
- [VSCode Extension Samples](https://github.com/microsoft/vscode-extension-samples)

---

## 🔄 持续维护

### 版本发布检查清单

- [ ] 更新版本号 (`npm version patch/minor/major`)
- [ ] 更新 README.md 版本历史
- [ ] 更新 README_zh.md 和 README_en.md
- [ ] 完整测试所有功能
- [ ] 测试语言切换
- [ ] 检查 API 调用
- [ ] 打包 (`npm run package`)
- [ ] 安装测试 VSIX
- [ ] 准备发布说明
- [ ] 分发给用户

### 代码审查要点

- [ ] 无 console.log 调试代码 (生产环境)
- [ ] 错误处理完善
- [ ] 国际化文本完整
- [ ] 性能优化 (无明显卡顿)
- [ ] 安全性检查 (API Key 不泄露)
- [ ] 文档同步更新

---

**当前版本**: 0.2.0
**最后更新**: 2025-11-28
**维护者**: EDA-Q Team

**祝你开发顺利! 🚀**
