# EDA-Q Assistant Development Documentation

> Complete VSCode Extension Development Guide - From Architecture to Release

---

## 📖 Table of Contents

- [Project Overview](#project-overview)
- [Technical Architecture](#technical-architecture)
- [Directory Structure](#directory-structure)
- [Core Files Explained](#core-files-explained)
- [Development Environment Setup](#development-environment-setup)
- [Development Workflow](#development-workflow)
- [Multilingual Implementation](#multilingual-implementation)
- [Packaging & Publishing](#packaging--publishing)
- [Common Development Issues](#common-development-issues)
- [Extension Development Guide](#extension-development-guide)

---

## 📋 Project Overview

### Project Information

- **Name**: EDA-Q Assistant
- **Type**: VSCode Extension (Plugin)
- **Version**: 0.2.0
- **Tech Stack**: Node.js + VSCode Extension API
- **AI Model**: Alibaba Cloud Qwen (extensible to other LLMs)
- **Code Size**: ~1000 lines of core code

### Features

✅ **Core Functionality**
- AI conversational code generation
- Context awareness (auto-read editor code)
- One-click code insertion to editor
- Streaming output simulation
- Conversation history management

✅ **Multilingual Support**
- Chinese/English interface switching
- Bilingual knowledge base (515 lines Chinese + 515 lines English API documentation)
- Dynamic language pack loading
- AI response language adaptation

✅ **Developer Friendly**
- Hot reload debugging
- Complete packaging scripts
- Automated version management
- Cross-platform support (Windows/Mac/Linux)

---

## 🏗️ Technical Architecture

### Overall Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                   VSCode Extension                   │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────────┐      ┌──────────────┐             │
│  │ extension.js│──────│ qwenClient.js│             │
│  │  (Entry)    │      │  (AI Client) │             │
│  └─────────────┘      └──────────────┘             │
│         │                      │                     │
│         │                      │                     │
│  ┌──────▼──────┐      ┌───────▼────────┐           │
│  │  Webview UI │      │  Qwen API      │           │
│  │  (Chat UI)  │      │  (Alibaba Qwen)│           │
│  └─────────────┘      └────────────────┘           │
│         │                                            │
│  ┌──────▼──────────────────────────┐               │
│  │  Language Packs (locales/)      │               │
│  │  Knowledge Base (knowledge/)    │               │
│  └─────────────────────────────────┘               │
└─────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| Extension Framework | VSCode Extension API | Official Extension API |
| Programming Language | JavaScript (Node.js) | Lightweight, easy to maintain |
| UI Framework | Webview (HTML/CSS/JS) | Native VSCode Webview |
| AI Service | Alibaba Cloud Qwen API | HTTP REST API |
| HTTP Client | Axios | Promise-based HTTP library |
| Internationalization | Custom i18n module | Lightweight, no dependencies |
| Packaging Tool | @vscode/vsce | Official packaging tool |

### Data Flow

```
User Input
   ↓
Webview UI (extension.js)
   ↓
Extract Context (current editor code)
   ↓
Build Prompt (system prompt + knowledge base + user question)
   ↓
QwenClient (qwenClient.js)
   ↓
Call Qwen API
   ↓
Receive AI Response
   ↓
Parse Code Blocks
   ↓
Webview Display (streaming simulation)
   ↓
User Actions (copy/insert code)
```

---

## 📁 Directory Structure

### Complete Directory Tree

```
edaq-assistant-mvp/
├── extension.js              # 🔥 Extension entry point (810 lines)
├── qwenClient.js             # 🔥 AI client (463 lines)
├── package.json              # 🔥 Extension configuration
├── LICENSE                   # MIT License
├── .vscodeignore            # Package exclusion config
├── icon.png                 # Extension icon (128x128)
│
├── locales/                 # 🌐 Language pack directory
│   ├── zh-CN.js            # Chinese language pack (UI text)
│   └── en-US.js            # English language pack (UI text)
│
├── knowledge/              # 📚 AI knowledge base
│   ├── user_manual_zh.txt  # Chinese API documentation (515 lines)
│   └── user_manual_en.txt  # English API documentation (515 lines)
│
├── scripts/                # 🛠️ Automation scripts
│   ├── package.ps1         # PowerShell packaging script
│   ├── package.bat         # Windows batch packaging
│   ├── package.sh          # Linux/Mac packaging script
│   ├── bump-version.ps1    # PowerShell version update
│   ├── bump-version.bat    # Windows version update
│   └── bump-version.sh     # Linux/Mac version update
│
├── node_modules/           # 📦 Dependencies (npm install)
│
├── .vscode/                # VSCode configuration
│   └── extensions.json     # Recommended extensions
│
├── README.md              # 📄 Project description (developer version)
├── README_zh.md           # 📄 User guide (Chinese)
├── README_en.md           # 📄 User guide (English)
│
├── demo_examples.py       # 🧪 Example code
├── start.bat             # Quick start script (Windows)
├── start.sh              # Quick start script (Linux/Mac)
│
└── edaq-assistant-0.2.0.vsix  # 📦 Packaged installation file
```

### Core File Descriptions

| File | Size | Purpose | Modification Frequency |
|------|------|---------|----------------------|
| `extension.js` | ~810 lines | Extension main logic and UI | High |
| `qwenClient.js` | ~463 lines | AI client and knowledge base | Medium |
| `package.json` | ~120 lines | Configuration and metadata | Medium |
| `locales/*.js` | ~150 lines/file | UI text translations | Low |
| `knowledge/*.txt` | ~515 lines/file | API documentation | Low |

---

## 🔥 Core Files Explained

### 1. extension.js - Extension Entry Point

**Responsibilities**: Extension lifecycle management, UI rendering, user interaction

#### Key Components

```javascript
// 1. Activation function - Extension entry
function activate(context) {
    // Load language pack
    // Register commands
    // Create WebviewProvider
    // Listen for configuration changes
}

// 2. ChatViewProvider - UI management class
class ChatViewProvider {
    constructor(context) {
        // Initialize language, conversation history
        // Listen for language switching
    }

    // Webview resolution and rendering
    resolveWebviewView(webviewView) {
        // Set HTML content
        // Listen for messages
    }

    // Process user messages
    async _processUserMessage(userMessage) {
        // Get configuration
        // Get context
        // Call AI
        // Return results
    }

    // Generate HTML content
    _getHtmlContent(webview) {
        // Render UI using language pack
        // Return complete HTML
    }
}
```

#### Core Flow

```javascript
// User message sending flow
User Input → sendMessage()
  ↓
Show thinking status
  ↓
_processUserMessage()
  ↓
Create QwenClient(apiKey, model, path, language)
  ↓
Call client.chat(message, context, history)
  ↓
Receive response
  ↓
Parse code blocks
  ↓
Streaming display (simulation)
  ↓
Add to conversation history
```

#### Important Methods

| Method | Parameters | Return Value | Purpose |
|--------|------------|--------------|---------|
| `activate()` | context | void | Extension activation entry |
| `loadLanguagePack()` | context, language | i18n | Load language pack |
| `_processUserMessage()` | userMessage | Promise | Process user message |
| `_getContext()` | - | {code, fileName} | Get editor context |
| `_insertCodeToEditor()` | code | void | Insert code to editor |
| `_getHtmlContent()` | webview | string | Generate UI HTML |

### 2. qwenClient.js - AI Client

**Responsibilities**: AI API calls, knowledge base management, prompt construction

#### Class Structure

```javascript
class QwenClient {
    constructor(apiKey, model, extensionPath, language) {
        this.apiKey = apiKey;
        this.model = model;  // qwen-plus/turbo/max
        this.language = language;  // zh-CN/en-US
        this.i18n = this._loadLanguagePack();
        this.knowledgeBase = this._loadKnowledgeBase();
    }

    // Load language pack
    _loadLanguagePack() { }

    // Load knowledge base
    _loadKnowledgeBase() { }

    // Build system prompt
    _buildSystemPrompt() { }

    // Main chat method
    async chat(userMessage, context, history) { }

    // Error handling
    _handleError(error) { }
}
```

#### System Prompt Structure

```javascript
_buildSystemPrompt() {
    return `
    You are a professional AI assistant for the EDA-Q quantum chip design tool.

    ## Core Knowledge Base
    ${this.knowledgeBase}  // Load language-specific API documentation

    ## Important Rules
    - Strictly follow knowledge base examples
    - Parameter names, types, and order must be correct
    - Wrap code with \`\`\`python
    - Use ${language} comments

    ## Typical Workflow
    [Contains complete code templates and sequencing rules]
    `;
}
```

#### API Call Flow

```javascript
// HTTP request structure
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

### 3. package.json - Extension Configuration

**Key Configuration Items**:

```json
{
  "name": "edaq-assistant",           // Extension ID
  "displayName": "EDA-Q Assistant",   // Display name
  "version": "0.2.0",                 // Version number
  "publisher": "edaq-team",           // Publisher

  "engines": {
    "vscode": "^1.80.0"              // Minimum VSCode version
  },

  "main": "./extension.js",           // Entry file

  "contributes": {
    "commands": [ ],                  // Registered commands
    "viewsContainers": { },          // Sidebar containers
    "views": { },                     // Views
    "configuration": { }              // Configuration items
  },

  "scripts": {
    "package": "vsce package",        // Package command
    "publish": "vsce publish"         // Publish command
  },

  "dependencies": {
    "axios": "^1.13.2"               // HTTP library
  }
}
```

### 4. Language Pack Structure (locales/)

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

### 5. Knowledge Base Structure (knowledge/)

**user_manual_zh.txt / user_manual_en.txt**:

```
# EDA-Q API Quick Reference

## 1. Design Class - Main Design Object
### Initialization
### Core Methods
  - generate_topology() - Generate topology
  - generate_qubits() - Generate qubits
  - generate_coupling_lines() - Generate couplers
  - ... (515 lines of detailed API documentation)

## 2. Topology Class
## 3. GDS Class
## 4. Complete Workflow Examples
## 5. Common Issues and Solutions
## 6. Design Best Practices
```

---

## 🛠️ Development Environment Setup

### Prerequisites

- **Node.js**: ≥ 16.0.0
- **npm**: ≥ 8.0.0
- **VSCode**: ≥ 1.80.0
- **Git**: For version control

### Quick Start

```bash
# 1. Clone/download project
cd edaq-assistant-mvp

# 2. Install dependencies
npm install

# 3. Open VSCode
code .

# 4. Press F5 to start debugging
# VSCode will open a new window with the extension loaded
```

### Dependencies Description

```json
{
  "dependencies": {
    "axios": "^1.13.2"          // HTTP request library
  },
  "devDependencies": {
    "@types/vscode": "^1.80.0", // VSCode API type definitions
    "eslint": "^8.50.0"         // Code linting
  }
}
```

### Debug Configuration

VSCode auto-generates `.vscode/launch.json`:

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

## 🔄 Development Workflow

### Daily Development Flow

```
Modify code
   ↓
Press F5 / Ctrl+R to reload
   ↓
Test in Extension Development Host
   ↓
View console output (debug logs)
   ↓
View Webview Developer Tools (UI debugging)
   ↓
Fix issues
   ↓
Repeat
```

### Debugging Tips

#### 1. Extension Host Debugging (extension.js)

```javascript
// Add breakpoints or console.log in code
console.log('✅ Extension activated');
console.log('📝 User message:', userMessage);
console.log('🔑 API Key:', apiKey ? 'Configured' : 'Not configured');
```

View output: **Debug Console**

#### 2. Webview UI Debugging

```javascript
// In Webview <script>
console.log('💬 Sending message:', message);
console.log('📦 Received response:', data);
```

View output:
1. In Extension Development Host window
2. Press `Ctrl+Shift+P`
3. Type "Developer: Open Webview Developer Tools"

#### 3. Network Request Debugging

```javascript
// qwenClient.js
console.log('🚀 Sending request to Qwen API...');
console.log('📝 Message count:', messages.length);
console.log('✅ Received API response');
console.log('📊 Token usage:', usage);
```

### Code Style

```javascript
// 1. Use clear comments
// User input validation
if (!message.trim()) return;

// 2. Use emoji for log marking
console.log('✅ Success');
console.log('❌ Error');
console.log('📝 Info');

// 3. Use async/await for async functions
async function processMessage() {
    try {
        const result = await client.chat();
    } catch (error) {
        console.error('❌ Error:', error);
    }
}

// 4. Comprehensive error handling
if (!apiKey) {
    throw new Error(i18n.errors.noApiKey);
}
```

---

## 🌐 Multilingual Implementation

### Architecture Design

```
Configuration (edaq.language: zh-CN/en-US)
    ↓
extension.js loads language pack
    ↓
QwenClient uses same language
    ↓
Webview UI renders with language pack
    ↓
Knowledge base loads language-specific documentation
    ↓
AI responds in corresponding language
```

### Implementation Details

#### 1. Language Pack Loading (extension.js)

```javascript
function loadLanguagePack(context, language = 'zh-CN') {
    const langFile = language === 'en-US' ? 'en-US.js' : 'zh-CN.js';
    const langPath = path.join(context.extensionPath, 'locales', langFile);

    // Clear cache
    delete require.cache[require.resolve(langPath)];
    return require(langPath);
}

// Listen for language changes
vscode.workspace.onDidChangeConfiguration(e => {
    if (e.affectsConfiguration('edaq.language')) {
        this._updateLanguage();
        // Re-render UI
        this._view.webview.html = this._getHtmlContent();
    }
});
```

#### 2. Knowledge Base Loading (qwenClient.js)

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

#### 3. UI Internationalization (extension.js)

```javascript
_getHtmlContent(webview) {
    const ui = this._i18n.ui;

    return `
        <h2>${ui.header.title}</h2>
        <p>${ui.header.subtitle}</p>

        <script>
            const i18n = ${JSON.stringify(ui)};

            // Use in JavaScript
            sendButton.textContent = i18n.input.sendButton;
        </script>
    `;
}
```

### Adding a New Language

```bash
# 1. Create language pack
cp locales/zh-CN.js locales/ja-JP.js

# 2. Translate all text
# Edit locales/ja-JP.js

# 3. Create knowledge base
cp knowledge/user_manual_zh.txt knowledge/user_manual_ja.txt

# 4. Translate API documentation
# Edit knowledge/user_manual_ja.txt

# 5. Update package.json
# Add "ja-JP" to enum and enumDescriptions

# 6. Update loading logic
# Add Japanese support in extension.js and qwenClient.js
```

---

## 📦 Packaging & Publishing

### Packaging Process

#### Method 1: Using Automation Scripts (Recommended)

**Windows PowerShell:**
```powershell
.\scripts\package.ps1
```

**Linux/Mac:**
```bash
./scripts/package.sh
```

**Features**:
1. ✅ Check Node.js and npm
2. ✅ Auto-install dependencies
3. ✅ Auto-install vsce
4. ✅ Execute packaging
5. ✅ Display generated file info

#### Method 2: Manual Packaging

```bash
# 1. Install vsce
npm install -g @vscode/vsce

# 2. Install dependencies
npm install

# 3. Package
vsce package

# Generated: edaq-assistant-0.2.0.vsix
```

### Version Management

#### Auto-update Version Number

```bash
# Patch version (0.2.0 -> 0.2.1) - Bug fixes
.\scripts\bump-version.ps1 patch

# Minor version (0.2.0 -> 0.3.0) - New features
.\scripts\bump-version.ps1 minor

# Major version (0.2.0 -> 1.0.0) - Breaking changes
.\scripts\bump-version.ps1 major
```

#### Manual Update

Edit `package.json`:
```json
{
  "version": "0.2.1"
}
```

### Publishing Process

```bash
# 1. Update version number
npm version patch  # or minor/major

# 2. Update documentation
# - README.md version history
# - README_zh.md version number
# - README_en.md version number

# 3. Re-package
npm run package

# 4. Test
# Install generated VSIX and fully test

# 5. Distribute
# - Email distribution
# - Internal server
# - GitHub Releases
```

### .vscodeignore Configuration

Exclude unnecessary files to reduce package size:

```
# Development files
.vscode/**
.vscode-test/**
node_modules/**/.bin/**

# Documentation (keep README.md)
*.md
!README.md

# Examples and tests
demo_examples.py
*.vsix

# Git files
.gitignore
.editorconfig
```

### Packaging Optimization

| Optimization | Method | Effect |
|--------------|--------|--------|
| Exclude dev files | .vscodeignore | Reduce 50% |
| Remove devDependencies | npm prune --production | Reduce 30% |
| Compress images | icon.png optimization | Reduce 10KB |

---

## ❓ Common Development Issues

### 1. Extension Won't Activate

**Symptom**: Extension doesn't show after pressing F5

**Troubleshooting**:
```javascript
// 1. Check activationEvents in package.json
"activationEvents": ["onStartupFinished"]

// 2. Check activate() function
function activate(context) {
    console.log('✅ Activation successful');
}

// 3. Check debug console for errors
```

### 2. Webview Shows Blank

**Cause**: CSP (Content Security Policy) or script errors

**Solution**:
```html
<!-- 1. Check CSP settings -->
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none';
               style-src 'unsafe-inline';
               script-src 'unsafe-inline' 'unsafe-eval';">

<!-- 2. Check JavaScript errors -->
<!-- Open Webview Developer Tools to check -->
```

### 3. API Call Failure

**Common Errors**:

```javascript
// Error 1: API Key not configured
if (!apiKey || apiKey.trim() === '') {
    throw new Error(this._i18n.errors.noApiKey);
}

// Error 2: Network issues
try {
    const response = await axios.post(url, data);
} catch (error) {
    if (error.code === 'ENOTFOUND') {
        // Network unreachable
    }
}

// Error 3: 401 Unauthorized
if (status === 401) {
    // API Key invalid
}
```

### 4. Language Switch Not Taking Effect

**Cause**: Cache not cleared

**Solution**:
```javascript
// Clear require cache when loading language pack
delete require.cache[require.resolve(langPath)];
return require(langPath);
```

### 5. Code Insertion Failure

**Cause**: Python file not open

**Solution**:
```javascript
const editor = vscode.window.activeTextEditor;
if (!editor) {
    vscode.window.showWarningMessage(
        this._i18n.ui.messages.openFileWarning
    );
    return;
}

// Optional: Check file type
if (editor.document.languageId !== 'python') {
    vscode.window.showWarningMessage('Please open a Python file');
    return;
}
```

### 6. Packaging Errors

**Error: Missing publisher**
```json
// package.json must have publisher
{
  "publisher": "edaq-team"
}
```

**Error: Icon not found**
```json
// Ensure icon.png exists, or remove config
{
  // "icon": "icon.png"  // Comment out
}
```

---

## 🚀 Extension Development Guide

### Adding New Features

#### Example: Adding "Code Explanation" Feature

**1. Add button in UI**:

```javascript
// extension.js - _getHtmlContent()
quickActions: [
    { icon: "📖", text: "Explain Code", question: "Explain what this code does" }
]
```

**2. Add handling logic**:

```javascript
// Existing _processUserMessage() will handle automatically
// Just ensure system prompt includes relevant instructions
```

**3. Update language packs**:

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

#### Example: Adding Configuration Item

**1. Add in package.json**:

```json
{
  "configuration": {
    "properties": {
      "edaq.maxTokens": {
        "type": "number",
        "default": 2000,
        "description": "Maximum token count"
      }
    }
  }
}
```

**2. Read in code**:

```javascript
// extension.js
const config = vscode.workspace.getConfiguration('edaq');
const maxTokens = config.get('maxTokens', 2000);

// Pass to QwenClient
const client = new QwenClient(apiKey, model, path, language, maxTokens);
```

### Switching AI Models

#### From Qwen to Other LLMs

**1. Create new AI client**:

```javascript
// openaiClient.js
class OpenAIClient {
    constructor(apiKey, model, extensionPath, language) {
        this.apiKey = apiKey;
        this.baseURL = 'https://api.openai.com/v1/chat/completions';
        // ...
    }

    async chat(userMessage, context, history) {
        // Implement OpenAI API call
    }
}
```

**2. Update configuration**:

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

**3. Dynamically select client**:

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

### Performance Optimization

#### 1. Reduce Package Size

```bash
# Install only production dependencies
npm install --production

# Check package size
vsce package --out test.vsix
du -h test.vsix
```

#### 2. Async Loading

```javascript
// Lazy load large dependencies
let heavyModule;

async function useHeavyFeature() {
    if (!heavyModule) {
        heavyModule = await import('./heavy-module');
    }
    return heavyModule.doSomething();
}
```

#### 3. Caching Optimization

```javascript
// Cache knowledge base content
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

## 📊 Code Statistics

### Core Code Size

| File | Lines | Description |
|------|-------|-------------|
| extension.js | 810 | Main logic and UI |
| qwenClient.js | 463 | AI client |
| locales/zh-CN.js | 150 | Chinese language pack |
| locales/en-US.js | 150 | English language pack |
| knowledge/user_manual_zh.txt | 515 | Chinese documentation |
| knowledge/user_manual_en.txt | 515 | English documentation |
| **Total** | **~2600** | **Core Code** |

### File Type Distribution

- **JavaScript**: 1423 lines (55%)
- **Text Documentation**: 1030 lines (40%)
- **JSON Configuration**: 120 lines (5%)

---

## 🎓 Best Practices

### 1. Code Organization

✅ **Recommended**:
```javascript
// Single responsibility
class QwenClient {
    // Only responsible for AI calls
}

class ChatViewProvider {
    // Only responsible for UI management
}
```

❌ **Avoid**:
```javascript
// All logic in activate()
function activate(context) {
    // 1000+ lines of code...
}
```

### 2. Error Handling

✅ **Recommended**:
```javascript
try {
    const result = await apiCall();
    return result;
} catch (error) {
    console.error('❌ API Error:', error);
    throw new Error(this.i18n.errors.networkError);
}
```

❌ **Avoid**:
```javascript
// Ignore errors
apiCall().catch(() => {});
```

### 3. User Experience

✅ **Recommended**:
```javascript
// Provide clear loading status
this._view.webview.postMessage({ type: 'thinking' });

// Provide error messages
vscode.window.showErrorMessage(i18n.errors.apiKeyInvalid);
```

❌ **Avoid**:
```javascript
// No feedback
await longRunningTask();
```

---

## 📚 Reference Resources

### Official Documentation

- [VSCode Extension API](https://code.visualstudio.com/api)
- [Webview API](https://code.visualstudio.com/api/extension-guides/webview)
- [Publishing Extensions](https://code.visualstudio.com/api/working-with-extensions/publishing-extension)

### Tools

- [@vscode/vsce](https://github.com/microsoft/vscode-vsce) - Packaging tool
- [Axios](https://axios-http.com/) - HTTP library
- [VSCode Extension Samples](https://github.com/microsoft/vscode-extension-samples)

---

## 🔄 Continuous Maintenance

### Version Release Checklist

- [ ] Update version number (`npm version patch/minor/major`)
- [ ] Update README.md version history
- [ ] Update README_zh.md and README_en.md
- [ ] Fully test all features
- [ ] Test language switching
- [ ] Check API calls
- [ ] Package (`npm run package`)
- [ ] Install and test VSIX
- [ ] Prepare release notes
- [ ] Distribute to users

### Code Review Checklist

- [ ] No console.log debug code (production environment)
- [ ] Complete error handling
- [ ] Complete internationalization text
- [ ] Performance optimization (no obvious lag)
- [ ] Security check (API Key not leaked)
- [ ] Documentation synchronized

---

**Current Version**: 0.2.0
**Last Updated**: 2025-11-28
**Maintainer**: EDA-Q Team

**Happy Coding! 🚀**
