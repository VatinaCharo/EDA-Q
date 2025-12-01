module.exports = {
    // UI界面文本
    ui: {
        header: {
            title: "🤖 EDA-Q AI Assistant",
            subtitle: "Powered by LLM | Quantum Chip Design Expert"
        },
        welcome: {
            icon: "🚀",
            title: "Welcome to EDA-Q AI Assistant!",
            description: "I can help you write quantum chip design code, answer technical questions, and optimize code structure",
            examplesTitle: "💡 Try these questions:",
            examples: [
                {
                    icon: "📊",
                    text: "How to design a 64-qubit superconducting quantum chip?",
                    question: "How to design a 64-qubit superconducting quantum chip?"
                },
                {
                    icon: "🔧",
                    text: "Generate complete code for a 4x4 topology",
                    question: "Generate complete code for a 4x4 topology"
                },
                {
                    icon: "🎯",
                    text: "How to add readout cavities?",
                    question: "How to add readout cavities?"
                },
                {
                    icon: "📚",
                    text: "What are the parameters of generate_topology?",
                    question: "What are the parameters of generate_topology?"
                }
            ]
        },
        quickActions: [
            { icon: "⚡", text: "Quick Start", question: "Generate complete workflow code" },
            { icon: "🔧", text: "Optimize", question: "Optimize this code" },
            { icon: "💡", text: "Explain", question: "Explain this code" },
            { icon: "🗑️", text: "Clear", action: "clear" }
        ],
        input: {
            placeholder: "Enter your question... (Shift+Enter for new line, Enter to send)",
            sendButton: "Send",
            sendingButton: "Sending..."
        },
        codeBlock: {
            copyButton: "📋 Copy",
            insertButton: "⬇️ Insert to Editor",
            copied: "✓ Copied"
        },
        thinking: {
            analyzing: "🤔 AI is analyzing your question",
            connecting: "🔗 Connecting to Qwen API",
            generating: "💡 Generating professional response"
        },
        messages: {
            errorPrefix: "❌ ",
            codeInserted: "✅ Code inserted",
            codeInsertFailed: "❌ Failed to insert code",
            openFileWarning: "Please open a Python file first",
            historyCleared: "Conversation history cleared",
            clearConfirm: "Are you sure to clear conversation history?"
        },
        notifications: {
            activated: "🤖 EDA-Q Assistant is ready!",
            openAssistant: "Open Assistant"
        }
    },

    // 错误消息
    errors: {
        noApiKey: "Please configure Qwen API Key in settings first\n\nSettings path: File → Preferences → Settings → Search \"EDA-Q\"",
        apiKeyInvalid: "API Key is invalid or expired\n\nPlease check:\n1. API Key is correct\n2. API Key is activated\n3. Account has sufficient balance",
        rateLimitExceeded: "Too many requests\n\nSuggestions:\n1. Wait a moment and retry\n2. Consider upgrading API plan",
        badRequest: "Request parameter error",
        timeout: "Request timeout\n\nPossible reasons:\n1. Unstable network connection\n2. Slow API service response\n\nPlease retry",
        networkError: "Cannot connect to API service\n\nPlease check:\n1. Network connection is normal\n2. Proxy configuration if needed",
        unknownError: "Unknown error",
        apiFormatError: "API response format error",
        retryOrContact: "Please retry or contact support"
    },

    // 配置项描述
    config: {
        apiKey: {
            description: "Alibaba Qwen API Key (Required)",
            markdownDescription: "Get API Key: [Alibaba Cloud Console](https://dashscope.console.aliyun.com/apiKey)"
        },
        model: {
            description: "Qwen Model Version",
            options: {
                plus: "Balanced - Best cost-effectiveness (Recommended)",
                turbo: "Fast - Faster response, lower cost",
                max: "Enhanced - Most powerful, higher cost"
            }
        },
        language: {
            description: "Interface Language / 界面语言",
            options: {
                zhCN: "简体中文",
                enUS: "English"
            }
        },
        enableContext: {
            description: "Automatically read code from editor as context"
        }
    }
};
