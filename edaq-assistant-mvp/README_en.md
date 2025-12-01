# EDA-Q Assistant User Guide

> Intelligent AI Code Assistant for EDA-Q Quantum Chip Design Tool

---

## 📖 Table of Contents

- [Quick Start](#quick-start)
- [Install Extension](#install-extension)
- [Configuration](#configuration)
- [Switch Language](#switch-language)
- [How to Use](#how-to-use)
- [FAQ](#faq)

---

## 🚀 Quick Start

### Three Steps to Get Started

1. **Install Extension** - Install VSIX file to VSCode
2. **Configure API Key** - Enter LLM API Key in settings
3. **Start Using** - Click sidebar icon and start chatting!

---

## 📦 Install Extension

### Step 1: Get Installation File

Get the `edaq-assistant-x.x.x.vsix` file from the `edaq-assistant-mvp` folder.

### Step 2: Install in VSCode

#### Method 1: Via Command Palette (Recommended)

1. Open VSCode
2. Press keyboard shortcut:
   - **Windows/Linux**: `Ctrl + Shift + P`
   - **Mac**: `Cmd + Shift + P`
3. Type: `Extensions: Install from VSIX...`
4. Select the downloaded `.vsix` file
5. Wait for installation to complete
6. Click **"Reload"** button or restart VSCode

#### Method 2: Via Extensions Panel

1. Open VSCode
2. Click **Extensions** icon on the left sidebar (or press `Ctrl+Shift+X`)
3. Click **"..."** menu in the top right corner
4. Select **"Install from VSIX..."**
5. Select the `.vsix` file
6. Reload VSCode

#### Method 3: Via Command Line

```bash
code --install-extension edaq-assistant-0.2.0.vsix
```

### Step 3: Verify Installation

After successful installation, you should see:
- ✅ **EDA-Q Assistant** icon appears in the left activity bar (circuit board icon)
- ✅ **EDA-Q Assistant** shows in extensions list
- ✅ "EDA-Q" related configurations can be found in settings

---

## ⚙️ Configuration

### Required Configuration: API Key

EDA-Q Assistant uses LLM API key. Here we use Alibaba Cloud Qwen as an example.

#### Step 1: Get API Key

1. Visit [Alibaba Cloud Console](https://dashscope.console.aliyun.com/apiKey)
2. Log in to your Alibaba Cloud account
3. Create or copy API Key

#### Step 2: Configure in VSCode

**Method 1: Via Settings Interface**

1. Open settings:
   - **Windows/Linux**: `File` → `Preferences` → `Settings`
   - **Mac**: `Code` → `Preferences` → `Settings`
   - Shortcut: `Ctrl+,` (Mac: `Cmd+,`)

2. Search: `EDA-Q` in the search box

3. Find **"Edaq: Qwen Api Key"**

4. Paste your API Key

**Method 2: Edit Configuration File**

1. Open command palette: `Ctrl+Shift+P` (Mac: `Cmd+Shift+P`)
2. Type: `Preferences: Open User Settings (JSON)`
3. Add the following configuration:

```json
{
  "edaq.qwenApiKey": "your-api-key-here"
}
```

### Optional Configuration

#### Select AI Model

Choose different Qwen models based on your needs:

- **qwen-plus** (default) - Balanced version, best cost-effectiveness, recommended
- **qwen-turbo** - Fast version, faster response, lower cost
- **qwen-max** - Enhanced version, most powerful, higher cost

Find **"Edaq: Qwen Model"** in settings to select.

#### Auto Read Code Context

- **Enabled** (default): AI will automatically read code from current editor as context
- **Disabled**: Only answers based on your question, without reading editor code

Find **"Edaq: Enable Context From Editor"** in settings to adjust.

---

## 🌐 Switch Language

EDA-Q Assistant supports bilingual interface in Chinese and English!

### Switching Steps

1. **Open Settings**: `Ctrl+,` (Mac: `Cmd+,`)

2. **Search**: Type `EDA-Q` in the search box

3. **Select Language**: Find **"Edaq: Language"** configuration

4. **Choose Interface Language**:
   - `简体中文` - Chinese interface and Chinese API documentation
   - `English` - English interface and English API documentation

5. **Reload**: Close and reopen EDA-Q Assistant sidebar
   - Or press `Ctrl+R` (Mac: `Cmd+R`) to reload window

### What Happens After Language Switch?

After switching language, the following will be automatically updated:

✅ **Interface Text**
- Title and welcome messages
- Quick action buttons
- Input placeholders
- Error messages

✅ **AI Knowledge Base**
- Automatically loads corresponding language EDA-Q API documentation
- Chinese mode: Uses Chinese API documentation
- English mode: Uses English API documentation

✅ **AI Response Language**
- AI assistant will respond in the selected language
- Generated code comments will also use the corresponding language

### Example Comparison

**Chinese Mode**:
```
Q: How to design a 64-qubit chip?
A: Let me help you generate the code...
    # 1. 创建Design对象
    design = Design()
```

**English Mode**:
```
Q: How to design a 64-qubit chip?
A: I'll help you generate the code...
   # 1. Create Design object
   design = Design()
```

---

## 💡 How to Use

### Launch Assistant

After installation and configuration:

1. Click **EDA-Q Assistant** icon in VSCode left activity bar
2. Or press `Ctrl+Shift+P` and type `EDA-Q: Open AI Assistant`

### Start Conversation

#### Method 1: Click Example Questions

When first opened, several example questions are displayed. Click to start quickly:

- 📊 How to design a 64-qubit superconducting quantum chip?
- 🔧 Generate complete code for a 4x4 topology
- 🎯 How to add readout cavities?
- 📚 What are the parameters of generate_topology?

#### Method 2: Type Your Question

Type your question directly in the bottom input box, for example:

```
Generate a 16-qubit quantum chip design code
```

#### Method 3: Use Quick Actions

Click quick action buttons:
- ⚡ **Quick Start** - Generate complete workflow code
- 🔧 **Optimize** - Optimize code in current editor
- 💡 **Explain** - Explain what the code does
- 🗑️ **Clear** - Clear conversation history

### Use Generated Code

When AI generates code, each code block has two buttons:

- **📋 Copy** - Copy code to clipboard
- **⬇️ Insert to Editor** - Directly insert to current editor cursor position

### Question Tips

#### ✅ Good Question Examples

```
Generate complete chip design code for 4x4 topology

How to add Transmon type qubits?

Parameter description of generate_topology method

What's wrong with this code? (ask after selecting code)
```

#### ❌ Not So Good Questions

```
Help me (too vague)

Code (no specific content)

How to do this (lack of context)
```

---

## ❓ FAQ

### Installation

#### Q1: Can't find extension icon after installation?

**A**: Try the following steps:
1. Restart VSCode (`Ctrl+R` or completely close and reopen)
2. Check extensions list, confirm EDA-Q Assistant is enabled
3. Check VSCode developer tools console for errors (`Help` → `Toggle Developer Tools`)

#### Q2: Prompt "Unable to install extension"?

**A**:
1. Ensure VSIX file is complete and not corrupted
2. Check if VSCode version is ≥ 1.80.0
3. Try command line installation: `code --install-extension xxx.vsix`

### Configuration

#### Q3: Still getting errors after configuring API Key?

**A**: Check the following:
1. API Key is copied correctly (no extra spaces)
2. API Key is activated on Alibaba Cloud
3. Alibaba Cloud account has sufficient balance
4. Network can access Alibaba Cloud services

#### Q4: How to view my configured API Key?

**A**:
1. Open settings: `Ctrl+,`
2. Search "EDA-Q"
3. View "Edaq: Qwen Api Key" configuration

### Usage

#### Q5: AI response is very slow?

**A**:
1. Switch to `qwen-turbo` model (faster)
2. Check network connection
3. Reduce code context length (disable auto-read or select less code)

#### Q6: AI generated code is not accurate?

**A**:
1. Provide more detailed question description
2. Provide code context (select relevant code in editor)
3. Try using `qwen-max` model (more powerful)

#### Q7: How to clear conversation history?

**A**:
Click **🗑️ Clear** button in quick actions.

#### Q8: Prompt "Please open a Python file first" when inserting code?

**A**:
1. First create or open a `.py` file
2. Place cursor where you want to insert code
3. Then click "Insert to Editor" button

### Language Switch

#### Q9: Interface doesn't change after switching language?

**A**:
1. Close EDA-Q Assistant sidebar and reopen
2. Or press `Ctrl+R` (Mac: `Cmd+R`) to reload VSCode window

#### Q10: Can I use mixed Chinese and English questions?

**A**:
Yes! AI will understand your question, but mainly respond in your set language.

### Updates

#### Q11: How to update to a new version?

**A**:
1. Get new `.vsix` file from development team
2. Follow installation steps to reinstall (will automatically overwrite old version)
3. Restart VSCode

#### Q12: Will updates lose my configuration?

**A**:
No! API Key and language settings will be preserved.

#### Q13: How to check current version?

**A**:
1. Open extensions panel: `Ctrl+Shift+X`
2. Search "EDA-Q Assistant"
3. View version number

---

## 📞 Get Help

### Having Issues?

1. Check [FAQ](#faq) section of this document
2. Check VSCode developer tools console for error messages
3. Contact development team for technical support

### Feedback

If you have any suggestions or find issues, feel free to contact the development team.

---

## 🎯 Quick Reference

### Common Shortcuts

| Action | Windows/Linux | Mac |
|--------|--------------|-----|
| Open Settings | `Ctrl+,` | `Cmd+,` |
| Command Palette | `Ctrl+Shift+P` | `Cmd+Shift+P` |
| Reload | `Ctrl+R` | `Cmd+R` |
| Open Extensions | `Ctrl+Shift+X` | `Cmd+Shift+X` |

### Configuration Quick Reference

| Config Item | Description | Default |
|-------------|-------------|---------|
| Language | Interface language | Simplified Chinese |
| Qwen Api Key | Qwen API key | (Required) |
| Qwen Model | AI model version | qwen-plus |
| Enable Context | Auto read code | Enabled |

---

**Current Version**: 0.2.0
**Last Updated**: 2025-11-28

**Enjoy using! 🎉**
