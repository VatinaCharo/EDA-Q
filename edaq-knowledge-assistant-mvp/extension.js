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
        this._isThinking = false; // 1. 初始化锁

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
        if (this._isThinking) {
            return; // 如果正在思考，直接忽略新的请求
        }

        this._isThinking = true; // 3. 加锁

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
            const client = new QwenClient({
                apiKey: apiKey,
                model: model,
                kbServiceUrl: 'http://localhost:5000',  // 知识库服务地址
                ragEnabled: true,
                ragTopK: 3
            }, this._context.extensionPath, this._language);

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
        }finally {
            this._isThinking = false; // 4. 无论成功失败，最后都要解锁
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
                currentCode = fullText.length > 6000
                    ? fullText.substring(0, 6000) + '\n# ... (代码过长,已截断)'
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
        // 1. 获取本地资源路径
        const scriptUri = webview.asWebviewUri(vscode.Uri.file(
            path.join(this._context.extensionPath, 'media', 'markdown-it.min.js')
        ));
        const highlightScriptUri = webview.asWebviewUri(vscode.Uri.file(
            path.join(this._context.extensionPath, 'media', 'highlight.min.js')
        ));
        const highlightStyleUri = webview.asWebviewUri(vscode.Uri.file(
            path.join(this._context.extensionPath, 'media', 'github-dark.min.css')
        ));

        const langAttr = this._language === 'en-US' ? 'en-US' : 'zh-CN';

        return `<!DOCTYPE html>
        <html lang="${langAttr}">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline' ${webview.cspSource}; script-src 'unsafe-inline' 'unsafe-eval' ${webview.cspSource};">
            <title>EDA-Q AI Assistant</title>
            <!-- 引入 Highlight.js 样式 -->
            <link href="${highlightStyleUri}" rel="stylesheet">
            <style>
                /* --- 基础重置 --- */
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body {
                    font-family: var(--vscode-font-family);
                    background: var(--vscode-editor-background);
                    color: var(--vscode-editor-foreground);
                    height: 100vh;
                    display: flex;
                    flex-direction: column;
                }

                /* --- 头部 --- */
                .header {
                    padding: 15px;
                    border-bottom: 1px solid var(--vscode-panel-border);
                    background: var(--vscode-sideBar-background);
                }
                .header h2 { font-size: 16px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
                .header p { font-size: 11px; color: var(--vscode-descriptionForeground); margin-top: 4px; }

                /* --- 消息区域 --- */
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
                /* 用户消息 */
                .user-message {
                    background: var(--vscode-inputValidation-infoBackground);
                    border-left: 3px solid var(--vscode-charts-blue);
                    align-self: flex-end;
                    white-space: pre-wrap; /* 用户输入保持换行 */
                }
                /* AI 消息 (Markdown 容器) */
                .assistant-message {
                    background: var(--vscode-editor-inactiveSelectionBackground);
                    border-left: 3px solid var(--vscode-charts-green);
                    align-self: flex-start;
                    /* 移除 pre-wrap，让 Markdown 库控制排版 */
                }
                .assistant-message p { margin-bottom: 8px; }
                .assistant-message p:last-child { margin-bottom: 0; }
                .assistant-message ul, .assistant-message ol { margin-left: 20px; margin-bottom: 8px; }
                .assistant-message strong { font-weight: bold; color: var(--vscode-textLink-foreground); }
                
                /* 错误消息 */
                .error-message {
                    background: var(--vscode-inputValidation-errorBackground);
                    border-left: 3px solid var(--vscode-charts-red);
                    color: var(--vscode-errorForeground);
                    align-self: flex-start;
                }

                /* --- 思考动画 --- */
                .thinking {
                    color: var(--vscode-descriptionForeground);
                    font-style: italic;
                    padding: 10px 12px;
                    background: var(--vscode-editor-inactiveSelectionBackground);
                    border-left: 3px solid var(--vscode-charts-yellow);
                    border-radius: 6px;
                    max-width: 90%;
                    align-self: flex-start;
                    animation: pulse 1.5s ease-in-out infinite;
                }
                @keyframes pulse { 0%, 100% { opacity: 0.6; } 50% { opacity: 1; } }

                /* --- 代码块样式 (保留你原来的设计) --- */
                .code-block {
                    margin: 10px 0;
                    border: 1px solid var(--vscode-panel-border);
                    border-radius: 6px;
                    overflow: hidden;
                    background: var(--vscode-textCodeBlock-background);
                }
                .code-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 6px 12px;
                    background: var(--vscode-editorGroupHeader-tabsBackground);
                    border-bottom: 1px solid var(--vscode-panel-border);
                }
                .code-lang {
                    font-size: 11px;
                    color: var(--vscode-descriptionForeground);
                    font-weight: 600;
                    text-transform: uppercase;
                }
                .code-actions { display: flex; gap: 6px; }
                .code-btn {
                    padding: 2px 8px;
                    font-size: 11px;
                    border: none;
                    border-radius: 3px;
                    cursor: pointer;
                    background: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                    transition: all 0.2s;
                }
                .code-btn:hover { background: var(--vscode-button-hoverBackground); }
                
                /* 代码内容区域 */
                .code-content {
                    padding: 10px;
                    overflow-x: auto;
                    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                }
                /* 覆盖 Highlight.js 可能自带的背景，使用 VSCode 主题色 */
                pre.hljs { background: transparent !important; padding: 0 !important; margin: 0 !important; }

                /* --- 输入框区域 --- */
                .input-container {
                    border-top: 1px solid var(--vscode-panel-border);
                    padding: 12px;
                    background: var(--vscode-sideBar-background);
                }
                .quick-actions { display: flex; gap: 6px; margin-bottom: 10px; flex-wrap: wrap; }
                .quick-btn {
                    padding: 6px 10px;
                    font-size: 11px;
                    border: 1px solid var(--vscode-button-border);
                    border-radius: 4px;
                    background: var(--vscode-button-secondaryBackground);
                    color: var(--vscode-button-secondaryForeground);
                    cursor: pointer;
                }
                .input-wrapper { display: flex; gap: 8px; align-items: flex-end; }
                #messageInput {
                    flex: 1; padding: 10px;
                    border: 1px solid var(--vscode-input-border);
                    border-radius: 4px;
                    background: var(--vscode-input-background);
                    color: var(--vscode-input-foreground);
                    font-size: 13px;
                    resize: none;
                    min-height: 38px; max-height: 120px;
                }
                #messageInput:focus { outline: none; border-color: var(--vscode-focusBorder); }
                #sendButton {
                    padding: 10px 18px; border: none; border-radius: 4px;
                    background: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                    cursor: pointer; font-weight: 600;
                }
                #sendButton:disabled { opacity: 0.5; cursor: not-allowed; }
                
                /* --- 欢迎页样式 --- */
                .welcome { text-align: center; padding: 30px 20px; color: var(--vscode-descriptionForeground); }
                .welcome-icon { font-size: 48px; margin-bottom: 15px; }
                .welcome-examples li {
                    padding: 10px; margin: 6px 0;
                    background: var(--vscode-list-hoverBackground);
                    border-radius: 4px; cursor: pointer;
                    list-style: none;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h2>${ui.header.title}</h2>
                <p>${ui.header.subtitle}</p>
            </div>

            <div class="messages" id="messages">
                <div class="welcome">
                    <div class="welcome-icon">${ui.welcome.icon}</div>
                    <h3>${ui.welcome.title}</h3>
                    <p>${ui.welcome.description}</p>
                    <div class="welcome-examples">
                        <ul>
                            ${ui.welcome.examples.map(ex =>
                                `<li onclick="askQuestion('${ex.question}')">${ex.icon} ${ex.text}</li>`
                            ).join('')}
                        </ul>
                    </div>
                </div>
            </div>

            <div class="input-container">
                <div class="quick-actions">
                    ${ui.quickActions.map(action =>
                        action.action === 'clear'
                            ? `<button class="quick-btn" onclick="clearChat()">${action.icon} ${action.text}</button>`
                            : `<button class="quick-btn" onclick="askQuestion('${action.question}')">${action.icon} ${action.text}</button>`
                    ).join('')}
                </div>
                <div class="input-wrapper">
                    <textarea id="messageInput" placeholder="${ui.input.placeholder}" rows="1"></textarea>
                    <button id="sendButton">${ui.input.sendButton}</button>
                </div>
            </div>

            <!-- 引入 Markdown 库 -->
            <script src="${scriptUri}"></script>
            <script src="${highlightScriptUri}"></script>

            <script>
                const i18n = ${JSON.stringify(ui)};
                const vscode = acquireVsCodeApi();
                const messagesDiv = document.getElementById('messages');
                const messageInput = document.getElementById('messageInput');
                const sendButton = document.getElementById('sendButton');

                // --- 1. 初始化 Markdown-it ---
                const md = window.markdownit({
                    html: true,
                    breaks: true,
                    linkify: true,
                    highlight: function (str, lang) {
                        // 使用 highlight.js 高亮代码
                        if (lang && hljs.getLanguage(lang)) {
                            try {
                                return hljs.highlight(str, { language: lang, ignoreIllegals: true }).value;
                            } catch (__) {}
                        }
                        return md.utils.escapeHtml(str);
                    }
                });

                // --- 2. 自定义代码块渲染 (保留你的按钮功能) ---
                // 重写 fence 规则：把代码包裹在 .code-block 结构中，并添加复制和插入按钮
                md.renderer.rules.fence = function (tokens, idx, options, env, self) {
                    const token = tokens[idx];
                    const info = token.info ? md.utils.unescapeAll(token.info).trim() : '';
                    const langName = info.split(/\\s+/)[0] || 'text';
                    const content = token.content; // 原始内容，用于复制和插入
                    
                    // 获取高亮后的 HTML
                    const highlighted = options.highlight(content, langName);

                    // 构造 HTML 结构 (注意这里把原始代码存在 data-code 属性中)
                    return \`
                    <div class="code-block" data-code="\${md.utils.escapeHtml(content)}">
                        <div class="code-header">
                            <span class="code-lang">\${langName}</span>
                            <div class="code-actions">
                                <button class="code-btn" onclick="copyCode(this)">\${i18n.codeBlock.copyButton}</button>
                                <button class="code-btn" onclick="insertCode(this)">\${i18n.codeBlock.insertButton}</button>
                            </div>
                        </div>
                        <div class="code-content">
                            <pre class="hljs"><code>\${highlighted}</code></pre>
                        </div>
                    </div>
                    \`;
                };

                // --- 消息处理 ---
                window.addEventListener('message', event => {
                    const message = event.data;
                    switch (message.type) {
                        case 'thinking':
                            addThinking();
                            break;

                        case 'responseStream': 
                        case 'responseFull': 
                            // 无论叫什么名字，只要后端发来的是完整 Markdown 文本
                            removeThinking();
                            enableInput();
                            
                            // 渲染 Markdown 为 HTML
                            const html = md.render(message.text);
                            
                            // 显示结果
                            addMessage(html, 'assistant', true);
                            break;

                        case 'error':
                            removeThinking();
                            enableInput();
                            addMessage(i18n.messages.errorPrefix + message.text, 'error');
                            break;
                    }
                });

                function enableInput() {
                    sendButton.disabled = false;
                    sendButton.textContent = i18n.input.sendButton;
                }

                // --- UI 辅助函数 ---
                
                // 添加消息 (支持 HTML)
                function addMessage(content, type, isHtml = false) {
                    const div = document.createElement('div');
                    div.className = 'message ' + type + '-message';
                    
                    if (isHtml) {
                        div.innerHTML = content; // 如果是 Markdown 渲染后的 HTML
                    } else {
                        div.textContent = content; // 如果是纯文本 (如用户提问)
                    }
                    
                    messagesDiv.appendChild(div);
                    // 移除欢迎页
                    const welcome = messagesDiv.querySelector('.welcome');
                    if (welcome) welcome.remove();
                    
                    scrollToBottom();
                }

                function addThinking() {
                    const div = document.createElement('div');
                    div.className = 'thinking';
                    div.id = 'thinking-indicator';
                    div.textContent = i18n.thinking.analyzing + '...';
                    messagesDiv.appendChild(div);
                    scrollToBottom();
                }

                function removeThinking() {
                    const el = document.getElementById('thinking-indicator');
                    if (el) el.remove();
                }

                function scrollToBottom() {
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                }

                // --- 用户交互 ---
                
                sendButton.addEventListener('click', sendMessage);
                messageInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                    }
                });

                function sendMessage() {
                    const text = messageInput.value.trim();
                    if (!text) return;

                    addMessage(text, 'user');
                    messageInput.value = '';
                    messageInput.style.height = 'auto';
                    
                    sendButton.disabled = true;
                    sendButton.textContent = i18n.input.sendingButton;

                    vscode.postMessage({ type: 'sendMessage', text: text });
                }

                function askQuestion(q) {
                    messageInput.value = q;
                    sendMessage();
                }

                function clearChat() {
                    if (confirm(i18n.messages.clearConfirm)) {
                        vscode.postMessage({ type: 'clearHistory' });
                        messagesDiv.innerHTML = '';
                    }
                }

                // --- 代码块功能实现 ---
                
                window.copyCode = function(btn) {
                    // 从父级 .code-block 的 data-code 属性中获取原始代码
                    const block = btn.closest('.code-block');
                    // data-code 是被 escape 过的，如果需要纯文本，浏览器会自动处理 dataset 取值，
                    // 但为了保险，我们通常直接用 dataset.code (它会自动解码 HTML 实体吗？不一定)
                    // 更稳妥的方式：
                    const code = block.getAttribute('data-code'); 
                    // 创建一个临时 textarea 来解码 HTML 实体
                    const txt = document.createElement("textarea");
                    txt.innerHTML = code;
                    const cleanCode = txt.value;

                    navigator.clipboard.writeText(cleanCode).then(() => {
                        const originalText = btn.textContent;
                        btn.textContent = i18n.codeBlock.copied;
                        setTimeout(() => { btn.textContent = originalText; }, 2000);
                    });
                };

                window.insertCode = function(btn) {
                    const block = btn.closest('.code-block');
                    const code = block.getAttribute('data-code');
                    const txt = document.createElement("textarea");
                    txt.innerHTML = code;
                    
                    vscode.postMessage({
                        type: 'insertCode',
                        code: txt.value
                    });
                };

                // 输入框高度自适应
                messageInput.addEventListener('input', function() {
                    this.style.height = 'auto';
                    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
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
