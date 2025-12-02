module.exports = {
    // UI界面文本
    ui: {
        header: {
            title: "🔬 Superconducting Quantum Computing Knowledge Assistant",
            subtitle: "Powered by LLM + RAG | Superconducting Quantum Computing Expert"
        },
        welcome: {
            icon: "🌟",
            title: "Welcome to Superconducting Quantum Computing Knowledge Assistant!",
            description: "I can answer Superconducting Quantum Computing questions, explain quantum concepts, and provide learning guidance based on professional knowledge base",
            examplesTitle: "💡 Try these questions:",
            examples: [
                {
                    icon: "⚛️",
                    text: "What is quantum entanglement?",
                    question: "What is quantum entanglement? Please explain in detail."
                },
                {
                    icon: "🔢",
                    text: "How does Shor's algorithm work?",
                    question: "How does Shor's algorithm achieve exponential speedup in factoring large numbers?"
                },
                {
                    icon: "🛡️",
                    text: "What is quantum error correction?",
                    question: "What is quantum error correction and why is it important?"
                },
                {
                    icon: "🔮",
                    text: "Difference between quantum and classical computing",
                    question: "What are the main differences between Superconducting Quantum Computing and classical computing?"
                }
            ]
        },
        quickActions: [
            { icon: "📖", text: "Basics", question: "Explain the basic principles of Superconducting Quantum Computing" },
            { icon: "🔍", text: "Deep Dive", question: "Please explain this concept in more detail" },
            { icon: "📝", text: "Summary", question: "Please summarize the key points" },
            { icon: "🗑️", text: "Clear", action: "clear" }
        ],
        input: {
            placeholder: "Enter your Superconducting Quantum Computing question... (Shift+Enter for new line, Enter to send)",
            sendButton: "Ask",
            sendingButton: "Thinking..."
        },
        codeBlock: {
            copyButton: "📋 Copy",
            saveButton: "💾 Save as Note",
            copied: "✓ Copied"
        },
        thinking: {
            analyzing: "🤔 AI is analyzing your question",
            searching: "🔍 Searching knowledge base",
            generating: "💡 Generating professional answer"
        },
        messages: {
            errorPrefix: "❌ ",
            noteSaved: "✅ Note saved",
            noteSaveFailed: "❌ Failed to save note",
            historyCleared: "Conversation history cleared",
            clearConfirm: "Are you sure to clear conversation history?"
        },
        notifications: {
            activated: "🔬 Quantum Knowledge Assistant is ready!",
            openAssistant: "Open Assistant"
        }
    },

    // 错误消息
    errors: {
        noApiKey: "Please configure API Key in settings first\n\nSettings path: File → Preferences → Settings → Search \"Quantum Assistant\"",
        apiKeyInvalid: "API Key is invalid or expired\n\nPlease check:\n1. API Key is correct\n2. API Key is activated\n3. Account has sufficient balance",
        rateLimitExceeded: "Too many requests\n\nSuggestions:\n1. Wait a moment and retry\n2. Consider upgrading API plan",
        badRequest: "Request parameter error",
        timeout: "Request timeout\n\nPossible reasons:\n1. Unstable network connection\n2. Slow API service response\n\nPlease retry",
        networkError: "Cannot connect to API service\n\nPlease check:\n1. Network connection is normal\n2. Proxy configuration if needed",
        unknownError: "Unknown error",
        apiFormatError: "API response format error",
        knowledgeBaseError: "Knowledge base loading error",
        embeddingError: "Text embedding error",
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
        enableRAG: {
            description: "Enable RAG (Retrieval-Augmented Generation) for more accurate answers"
        },
        topK: {
            description: "Number of relevant documents to retrieve (1-10)"
        }
    },

    // 知识领域分类
    knowledgeDomains: {
        basics: "Superconducting Quantum Computing Basics",
        algorithms: "Quantum Algorithms",
        hardware: "Quantum Hardware",
        errorCorrection: "Quantum Error Correction",
        applications: "Quantum Applications",
        cryptography: "Quantum Cryptography"
    }
};
