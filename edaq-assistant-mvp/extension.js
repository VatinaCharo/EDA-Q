const vscode = require('vscode');
const QwenClient = require('./qwenClient');
const path = require('path');

/**
 * 加载语言包
 */
function loadLanguagePack(context, language = 'zh-CN') {
    try {
        const langFile = language === 'en-US' ? 'en-US.js' : 'zh-CN.js';
        const langPath = path.join(context.extensionPath, 'locales', langFile);
        // 清除缓存以确保重新加载
        delete require.cache[require.resolve(langPath)];
        return require(langPath);
    } catch (error) {
        console.error('加载语言包失败:', error);
        return require(path.join(context.extensionPath, 'locales', 'zh-CN.js'));
    }
}

/**
 * 激活扩展
 */
function activate(context) {
    const config = vscode.workspace.getConfiguration('edaq');
    const language = config.get('language', 'zh-CN');
    const i18n = loadLanguagePack(context, language);

    console.log('✅ EDA-Q Assistant 已激活');

    // 注册命令
    let openChatCommand = vscode.commands.registerCommand('edaq.openChat', () => {
        vscode.commands.executeCommand('edaq.chatView.focus');
    });

    context.subscriptions.push(openChatCommand);

    // 注册 Webview Provider
    const provider = new ChatViewProvider(context);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('edaq.chatView', provider)
    );

    // 启动时显示欢迎消息
    setTimeout(() => {
        vscode.window.showInformationMessage(
            i18n.ui.notifications.activated,
            i18n.ui.notifications.openAssistant
        ).then(selection => {
            if (selection === i18n.ui.notifications.openAssistant) {
                vscode.commands.executeCommand('edaq.openChat');
            }
        });
    }, 1000);
}

/**
 * Webview Provider - 管理聊天界面
 */
class ChatViewProvider {
    constructor(context) {
        this._context = context;
        this._view = null;
        this._conversationHistory = [];
        this._updateLanguage();

        // 监听配置变化
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('edaq.language')) {
                this._updateLanguage();
                if (this._view) {
                    this._view.webview.html = this._getHtmlContent(this._view.webview);
                }
            }
        });
    }

    _updateLanguage() {
        const config = vscode.workspace.getConfiguration('edaq');
        this._language = config.get('language', 'zh-CN');
        this._i18n = loadLanguagePack(this._context, this._language);
    }

    resolveWebviewView(webviewView) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._context.extensionUri]
        };

        webviewView.webview.html = this._getHtmlContent(webviewView.webview);

        // 处理来自webview的消息
        webviewView.webview.onDidReceiveMessage(async (message) => {
            await this._handleMessage(message);
        });
    }

    async _handleMessage(message) {
        switch (message.type) {
            case 'sendMessage':
                await this._processUserMessage(message.text);
                break;
            case 'insertCode':
                this._insertCodeToEditor(message.code);
                break;
            case 'clearHistory':
                this._conversationHistory = [];
                vscode.window.showInformationMessage(this._i18n.ui.messages.historyCleared);
                break;
        }
    }

    async _processUserMessage(userMessage) {
        // 显示加载状态
        this._view.webview.postMessage({ type: 'thinking' });

        try {
            // 获取配置
            const config = vscode.workspace.getConfiguration('edaq');
            const apiKey = config.get('qwenApiKey');
            const model = config.get('qwenModel');
            const enableContext = config.get('enableContextFromEditor');

            if (!apiKey || apiKey.trim() === '') {
                throw new Error(this._i18n.errors.noApiKey);
            }

            // 获取上下文
            const context = enableContext ? this._getContext() : {};

            // 创建客户端并调用API
            const client = new QwenClient(apiKey, model, this._context.extensionPath, this._language);

            // 不使用真实SSE流，使用普通模式
            const response = await client.chat(userMessage, context, this._conversationHistory);

            // 保存到对话历史
            this._conversationHistory.push(
                { role: 'user', content: userMessage },
                { role: 'assistant', content: response.text }
            );

            // 发送完整响应，让前端模拟流式显示
            this._view.webview.postMessage({
                type: 'responseStream',
                text: response.text,
                code: response.code
            });

        } catch (error) {
            console.error('处理消息失败:', error);
            this._view.webview.postMessage({
                type: 'error',
                text: error.message || '发生未知错误'
            });
        }
    }

    _getContext() {
        const editor = vscode.window.activeTextEditor;
        let currentCode = '';
        let fileName = '';

        if (editor && editor.document.languageId === 'python') {
            fileName = editor.document.fileName;
            const selection = editor.selection;

            if (!selection.isEmpty) {
                // 如果有选中代码,使用选中部分
                currentCode = editor.document.getText(selection);
            } else {
                // 否则使用整个文档(限制长度)
                const fullText = editor.document.getText();
                currentCode = fullText.length > 5000
                    ? fullText.substring(0, 5000) + '\n# ... (代码过长,已截断)'
                    : fullText;
            }
        }

        return {
            currentCode: currentCode,
            fileName: fileName
        };
    }

    _insertCodeToEditor(code) {
        const editor = vscode.window.activeTextEditor;

        if (!editor) {
            vscode.window.showWarningMessage(this._i18n.ui.messages.openFileWarning);
            return;
        }

        editor.edit(editBuilder => {
            editBuilder.insert(editor.selection.active, code);
        }).then(success => {
            if (success) {
                vscode.window.showInformationMessage(this._i18n.ui.messages.codeInserted);
            } else {
                vscode.window.showErrorMessage(this._i18n.ui.messages.codeInsertFailed);
            }
        });
    }

    _getHtmlContent(webview) {
        const ui = this._i18n.ui;
        const langAttr = this._language === 'en-US' ? 'en-US' : 'zh-CN';

        return `<!DOCTYPE html>
        <html lang="${langAttr}">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline' 'unsafe-eval';">
            <title>EDA-Q AI Assistant</title>
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: var(--vscode-editor-background);
                    color: var(--vscode-editor-foreground);
                    height: 100vh;
                    display: flex;
                    flex-direction: column;
                }
                .header {
                    padding: 15px;
                    border-bottom: 1px solid var(--vscode-panel-border);
                    background: var(--vscode-sideBar-background);
                }
                .header h2 {
                    font-size: 16px;
                    font-weight: 600;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                .header p {
                    font-size: 11px;
                    color: var(--vscode-descriptionForeground);
                    margin-top: 4px;
                }
                .messages {
                    flex: 1;
                    overflow-y: auto;
                    padding: 15px;
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                }
                .message {
                    padding: 10px 12px;
                    border-radius: 6px;
                    line-height: 1.6;
                    font-size: 13px;
                    max-width: 90%;
                    word-wrap: break-word;
                }
                .user-message {
                    background: var(--vscode-inputValidation-infoBackground);
                    border-left: 3px solid var(--vscode-charts-blue);
                    align-self: flex-end;
                }
                .assistant-message {
                    background: var(--vscode-editor-inactiveSelectionBackground);
                    border-left: 3px solid var(--vscode-charts-green);
                    align-self: flex-start;
                    white-space: pre-wrap;
                }
                .error-message {
                    background: var(--vscode-inputValidation-errorBackground);
                    border-left: 3px solid var(--vscode-charts-red);
                    color: var(--vscode-errorForeground);
                    align-self: flex-start;
                    white-space: pre-line;
                }
                .thinking {
                    text-align: left;
                    color: var(--vscode-descriptionForeground);
                    font-style: italic;
                    padding: 10px 12px;
                    background: var(--vscode-editor-inactiveSelectionBackground);
                    border-left: 3px solid var(--vscode-charts-yellow);
                    border-radius: 6px;
                    max-width: 90%;
                    align-self: flex-start;
                    animation: pulse 1.5s ease-in-out infinite;
                    white-space: pre-wrap;
                }
                @keyframes pulse {
                    0%, 100% { opacity: 0.6; }
                    50% { opacity: 1; }
                }
                .code-block {
                    background: var(--vscode-textCodeBlock-background);
                    border: 1px solid var(--vscode-panel-border);
                    border-radius: 6px;
                    margin: 10px 0;
                    overflow: visible;
                    width: 100%;
                    align-self: flex-start;
                }
                .code-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 8px 12px;
                    background: var(--vscode-editorGroupHeader-tabsBackground);
                    border-bottom: 1px solid var(--vscode-panel-border);
                }
                .code-lang {
                    font-size: 11px;
                    color: var(--vscode-descriptionForeground);
                    font-weight: 600;
                    text-transform: uppercase;
                }
                .code-actions {
                    display: flex;
                    gap: 6px;
                }
                .code-btn {
                    padding: 4px 10px;
                    font-size: 11px;
                    border: none;
                    border-radius: 3px;
                    cursor: pointer;
                    background: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                    transition: all 0.2s;
                    font-weight: 500;
                }
                .code-btn:hover {
                    background: var(--vscode-button-hoverBackground);
                    transform: translateY(-1px);
                }
                .code-btn:active {
                    transform: translateY(0);
                }
                .code-content {
                    padding: 12px;
                    overflow-x: auto;
                    max-height: 600px;
                    overflow-y: auto;
                    min-height: 50px;
                    background: var(--vscode-textCodeBlock-background);
                }
                pre {
                    margin: 0;
                    padding: 0;
                    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                    font-size: 13px;
                    line-height: 1.6;
                    white-space: pre;
                    overflow-wrap: normal;
                    min-height: 30px;
                }
                code {
                    display: block;
                    font-family: inherit;
                }
                .input-container {
                    border-top: 1px solid var(--vscode-panel-border);
                    padding: 12px;
                    background: var(--vscode-sideBar-background);
                }
                .quick-actions {
                    display: flex;
                    gap: 6px;
                    margin-bottom: 10px;
                    flex-wrap: wrap;
                }
                .quick-btn {
                    padding: 6px 10px;
                    font-size: 11px;
                    border: 1px solid var(--vscode-button-border);
                    border-radius: 4px;
                    background: var(--vscode-button-secondaryBackground);
                    color: var(--vscode-button-secondaryForeground);
                    cursor: pointer;
                    transition: all 0.2s;
                    white-space: nowrap;
                }
                .quick-btn:hover {
                    background: var(--vscode-button-secondaryHoverBackground);
                    transform: translateY(-1px);
                }
                .input-wrapper {
                    display: flex;
                    gap: 8px;
                    align-items: flex-end;
                }
                #messageInput {
                    flex: 1;
                    padding: 10px;
                    border: 1px solid var(--vscode-input-border);
                    border-radius: 4px;
                    background: var(--vscode-input-background);
                    color: var(--vscode-input-foreground);
                    font-size: 13px;
                    resize: none;
                    min-height: 38px;
                    max-height: 120px;
                    font-family: inherit;
                }
                #messageInput:focus {
                    outline: none;
                    border-color: var(--vscode-focusBorder);
                }
                #sendButton {
                    padding: 10px 18px;
                    border: none;
                    border-radius: 4px;
                    background: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                    cursor: pointer;
                    font-size: 13px;
                    font-weight: 600;
                    transition: all 0.2s;
                    white-space: nowrap;
                }
                #sendButton:hover:not(:disabled) {
                    background: var(--vscode-button-hoverBackground);
                    transform: translateY(-1px);
                }
                #sendButton:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                }
                .welcome {
                    text-align: center;
                    padding: 30px 20px;
                    color: var(--vscode-descriptionForeground);
                }
                .welcome-icon {
                    font-size: 48px;
                    margin-bottom: 15px;
                }
                .welcome h3 {
                    margin-bottom: 10px;
                    color: var(--vscode-foreground);
                    font-size: 16px;
                }
                .welcome p {
                    margin-bottom: 20px;
                    font-size: 13px;
                }
                .welcome-examples {
                    text-align: left;
                    margin-top: 20px;
                }
                .welcome-examples h4 {
                    margin-bottom: 10px;
                    font-size: 12px;
                    color: var(--vscode-foreground);
                    font-weight: 600;
                }
                .welcome-examples ul {
                    list-style: none;
                }
                .welcome-examples li {
                    padding: 10px;
                    margin: 6px 0;
                    background: var(--vscode-list-hoverBackground);
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                    transition: all 0.2s;
                    border-left: 3px solid transparent;
                }
                .welcome-examples li:hover {
                    background: var(--vscode-list-activeSelectionBackground);
                    border-left-color: var(--vscode-charts-blue);
                    transform: translateX(5px);
                }
                .scrollbar::-webkit-scrollbar {
                    width: 8px;
                }
                .scrollbar::-webkit-scrollbar-track {
                    background: var(--vscode-scrollbarSlider-background);
                }
                .scrollbar::-webkit-scrollbar-thumb {
                    background: var(--vscode-scrollbarSlider-hoverBackground);
                    border-radius: 4px;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h2>${ui.header.title}</h2>
                <p>${ui.header.subtitle}</p>
            </div>

            <div class="messages scrollbar" id="messages">
                <div class="welcome">
                    <div class="welcome-icon">${ui.welcome.icon}</div>
                    <h3>${ui.welcome.title}</h3>
                    <p>${ui.welcome.description}</p>
                    <div class="welcome-examples">
                        <h4>${ui.welcome.examplesTitle}</h4>
                        <ul>
                            ${ui.welcome.examples.map(ex =>
                                `<li onclick="askQuestion('${ex.question}')">
                                    ${ex.icon} ${ex.text}
                                </li>`
                            ).join('')}
                        </ul>
                    </div>
                </div>
            </div>

            <div class="input-container">
                <div class="quick-actions">
                    ${ui.quickActions.map(action =>
                        action.action === 'clear'
                            ? `<button class="quick-btn" onclick="clearChat()">
                                ${action.icon} ${action.text}
                            </button>`
                            : `<button class="quick-btn" onclick="askQuestion('${action.question}')">
                                ${action.icon} ${action.text}
                            </button>`
                    ).join('')}
                </div>
                <div class="input-wrapper">
                    <textarea
                        id="messageInput"
                        placeholder="${ui.input.placeholder}"
                        rows="1"
                    ></textarea>
                    <button id="sendButton">${ui.input.sendButton}</button>
                </div>
            </div>

            <script>
                // Language pack
                const i18n = ${JSON.stringify(ui)};

                const vscode = acquireVsCodeApi();
                const messagesDiv = document.getElementById('messages');
                const messageInput = document.getElementById('messageInput');
                const sendButton = document.getElementById('sendButton');

                // 自动调整输入框高度
                messageInput.addEventListener('input', function() {
                    this.style.height = 'auto';
                    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
                });

                // 发送消息
                sendButton.addEventListener('click', sendMessage);
                messageInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                    }
                });

                function sendMessage() {
                    const message = messageInput.value.trim();
                    if (!message) return;

                    // 清除欢迎消息
                    const welcome = messagesDiv.querySelector('.welcome');
                    if (welcome) welcome.remove();

                    // 显示用户消息
                    addMessage(message, 'user');
                    messageInput.value = '';
                    messageInput.style.height = 'auto';

                    // 禁用发送按钮
                    sendButton.disabled = true;
                    sendButton.textContent = i18n.input.sendingButton;

                    // 发送到扩展
                    vscode.postMessage({
                        type: 'sendMessage',
                        text: message
                    });
                }

                function askQuestion(question) {
                    messageInput.value = question;
                    sendMessage();
                }

                function addMessage(text, sender) {
                    const messageDiv = document.createElement('div');
                    messageDiv.className = 'message ' + sender + '-message';
                    messageDiv.textContent = text;
                    messagesDiv.appendChild(messageDiv);
                    scrollToBottom();
                }

                function addThinking() {
                    const thinkingDiv = document.createElement('div');
                    thinkingDiv.className = 'thinking';
                    thinkingDiv.id = 'thinking-indicator';
                    thinkingDiv.innerHTML = i18n.thinking.analyzing + '...<br>' + i18n.thinking.connecting + '...';
                    messagesDiv.appendChild(thinkingDiv);
                    scrollToBottom();

                    let dots = 0;
                    const interval = setInterval(function() {
                        const thinking = document.getElementById('thinking-indicator');
                        if (!thinking) {
                            clearInterval(interval);
                            return;
                        }
                        dots = (dots + 1) % 4;
                        const dotStr = '.'.repeat(dots);
                        thinking.innerHTML = i18n.thinking.analyzing + dotStr + '<br>' + i18n.thinking.generating + dotStr;
                    }, 500);
                    thinkingDiv.dataset.interval = interval;
                }

                function removeThinking() {
                    const thinking = document.getElementById('thinking-indicator');
                    if (thinking) {
                        if (thinking.dataset.interval) {
                            clearInterval(parseInt(thinking.dataset.interval));
                        }
                        thinking.remove();
                    }
                }

                function addCodeBlock(code, language = 'python') {
                    const codeDiv = document.createElement('div');
                    codeDiv.className = 'code-block';
                    codeDiv.innerHTML = '<div class="code-header">' +
                        '<span class="code-lang">' + language + '</span>' +
                        '<div class="code-actions">' +
                        '<button class="code-btn" onclick="copyCode(this)">' + i18n.codeBlock.copyButton + '</button>' +
                        '<button class="code-btn" onclick="insertCode(this)">' + i18n.codeBlock.insertButton + '</button>' +
                        '</div>' +
                        '</div>' +
                        '<div class="code-content">' +
                        '<pre><code>' + escapeHtml(code) + '</code></pre>' +
                        '</div>';
                    codeDiv.dataset.code = code;
                    messagesDiv.appendChild(codeDiv);
                    scrollToBottom();
                }

                function copyCode(btn) {
                    const code = btn.closest('.code-block').dataset.code;
                    navigator.clipboard.writeText(code).then(() => {
                        const originalText = btn.textContent;
                        btn.textContent = i18n.codeBlock.copied;
                        setTimeout(() => {
                            btn.textContent = originalText;
                        }, 2000);
                    });
                }

                function insertCode(btn) {
                    const code = btn.closest('.code-block').dataset.code;
                    vscode.postMessage({
                        type: 'insertCode',
                        code: code
                    });
                }

                function clearChat() {
                    if (confirm(i18n.messages.clearConfirm)) {
                        vscode.postMessage({ type: 'clearHistory' });
                        messagesDiv.innerHTML = '';
                        location.reload();
                    }
                }

                function escapeHtml(text) {
                    const div = document.createElement('div');
                    div.textContent = text;
                    return div.innerHTML;
                }

                function scrollToBottom() {
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                }

                // 流式输出相关变量
                let streamingDiv = null;
                let streamingText = '';

                // 接收来自扩展的消息
                window.addEventListener('message', event => {
                    const message = event.data;

                    switch (message.type) {
                        case 'thinking':
                            addThinking();
                            streamingDiv = null;
                            streamingText = '';
                            break;

                        case 'responseStream':
                            removeThinking();

                            var fullText = message.text;
                            var charIndex = 0;
                            var tempDiv = document.createElement('div');
                            tempDiv.className = 'message assistant-message';
                            tempDiv.id = 'streaming-message';
                            messagesDiv.appendChild(tempDiv);

                            var streamInterval = setInterval(function() {
                                if (charIndex < fullText.length) {
                                    var chunkSize = Math.min(3, fullText.length - charIndex);
                                    tempDiv.textContent += fullText.substr(charIndex, chunkSize);
                                    charIndex += chunkSize;
                                    scrollToBottom();
                                } else {
                                    clearInterval(streamInterval);
                                    tempDiv.remove();

                                    sendButton.disabled = false;
                                    sendButton.textContent = i18n.input.sendButton;

                                    var backtick = String.fromCharCode(96);
                                    var codeRegex = new RegExp(backtick + backtick + backtick + '(\\\\w*)\\\\n([\\\\s\\\\S]*?)' + backtick + backtick + backtick, 'g');
                                    var lastIndex = 0;
                                    var match;
                                    var hasCode = false;

                                    while ((match = codeRegex.exec(fullText)) !== null) {
                                        hasCode = true;
                                        var textBefore = fullText.slice(lastIndex, match.index).trim();
                                        if (textBefore) {
                                            addMessage(textBefore, 'assistant');
                                        }

                                        var language = match[1] || 'python';
                                        var code = match[2];
                                        addCodeBlock(code, language);

                                        lastIndex = match.index + match[0].length;
                                    }

                                    var textAfter = fullText.slice(lastIndex).trim();
                                    if (textAfter) {
                                        addMessage(textAfter, 'assistant');
                                    }

                                    if (!hasCode) {
                                        addMessage(fullText, 'assistant');
                                    }
                                }
                            }, 30);
                            break;

                        case 'response':
                            removeThinking();
                            sendButton.disabled = false;
                            sendButton.textContent = i18n.input.sendButton;

                            if (message.text) {
                                const backtick = String.fromCharCode(96);
                                const codeRegex = new RegExp(backtick + backtick + backtick + '(\\\\w*)\\\\n([\\\\s\\\\S]*?)' + backtick + backtick + backtick, 'g');
                                let lastIndex = 0;
                                let match;
                                let hasCode = false;

                                while ((match = codeRegex.exec(message.text)) !== null) {
                                    hasCode = true;
                                    const textBefore = message.text.slice(lastIndex, match.index).trim();
                                    if (textBefore) {
                                        addMessage(textBefore, 'assistant');
                                    }

                                    const language = match[1] || 'python';
                                    const code = match[2];
                                    addCodeBlock(code, language);

                                    lastIndex = match.index + match[0].length;
                                }

                                const textAfter = message.text.slice(lastIndex).trim();
                                if (textAfter) {
                                    addMessage(textAfter, 'assistant');
                                }

                                if (!hasCode) {
                                    addMessage(message.text, 'assistant');
                                }
                            }
                            break;

                        case 'error':
                            removeThinking();
                            sendButton.disabled = false;
                            sendButton.textContent = i18n.input.sendButton;
                            const errorDiv = document.createElement('div');
                            errorDiv.className = 'message error-message';
                            errorDiv.textContent = i18n.messages.errorPrefix + message.text;
                            messagesDiv.appendChild(errorDiv);
                            scrollToBottom();
                            break;
                    }
                });
            </script>
        </body>
        </html>`;
    }
}

function deactivate() {
    console.log('EDA-Q Assistant 已停用');
}

module.exports = {
    activate,
    deactivate
};
