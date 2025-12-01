module.exports = {
    // UI界面文本
    ui: {
        header: {
            title: "🤖 EDA-Q AI 助手",
            subtitle: "基于LLM | 量子芯片设计专家"
        },
        welcome: {
            icon: "🚀",
            title: "欢迎使用 EDA-Q AI 助手!",
            description: "我可以帮助您编写量子芯片设计代码、解答技术问题、优化代码结构",
            examplesTitle: "💡 试试这些问题:",
            examples: [
                {
                    icon: "📊",
                    text: "如何设计一个64比特的超导量子芯片?",
                    question: "如何设计一个64比特的超导量子芯片?"
                },
                {
                    icon: "🔧",
                    text: "生成一个4x4拓扑结构的完整代码",
                    question: "生成一个4x4拓扑结构的完整代码"
                },
                {
                    icon: "🎯",
                    text: "如何添加读取谐振腔?",
                    question: "如何添加读取谐振腔?"
                },
                {
                    icon: "📚",
                    text: "generate_topology有哪些参数?",
                    question: "generate_topology有哪些参数?"
                }
            ]
        },
        quickActions: [
            { icon: "⚡", text: "快速开始", question: "生成完整流程代码" },
            { icon: "🔧", text: "优化代码", question: "优化这段代码" },
            { icon: "💡", text: "解释代码", question: "解释这段代码" },
            { icon: "🗑️", text: "清空", action: "clear" }
        ],
        input: {
            placeholder: "输入你的问题... (Shift+Enter 换行, Enter 发送)",
            sendButton: "发送",
            sendingButton: "发送中..."
        },
        codeBlock: {
            copyButton: "📋 复制",
            insertButton: "⬇️ 插入编辑器",
            copied: "✓ 已复制"
        },
        thinking: {
            analyzing: "🤔 AI正在分析您的问题",
            connecting: "🔗 连接千问API中",
            generating: "💡 生成专业回复中"
        },
        messages: {
            errorPrefix: "❌ ",
            codeInserted: "✅ 代码已插入",
            codeInsertFailed: "❌ 代码插入失败",
            openFileWarning: "请先打开一个Python文件",
            historyCleared: "对话历史已清空",
            clearConfirm: "确定要清空对话历史吗?"
        },
        notifications: {
            activated: "🤖 EDA-Q Assistant 已就绪!",
            openAssistant: "打开助手"
        }
    },

    // 错误消息
    errors: {
        noApiKey: "请先在设置中配置千问 API Key\n\n设置路径: 文件 → 首选项 → 设置 → 搜索 \"EDA-Q\"",
        apiKeyInvalid: "API Key无效或已过期\n\n请检查:\n1. API Key是否正确\n2. API Key是否已激活\n3. 账户余额是否充足",
        rateLimitExceeded: "请求过于频繁\n\n建议:\n1. 稍等片刻再试\n2. 考虑升级API套餐",
        badRequest: "请求参数错误",
        timeout: "请求超时\n\n可能原因:\n1. 网络连接不稳定\n2. API服务响应慢\n\n建议重试",
        networkError: "无法连接到API服务\n\n请检查:\n1. 网络连接是否正常\n2. 是否需要配置代理",
        unknownError: "未知错误",
        apiFormatError: "API返回格式异常",
        retryOrContact: "请重试或联系技术支持"
    },

    // 配置项描述
    config: {
        apiKey: {
            description: "阿里云千问 API Key (必填)",
            markdownDescription: "获取API Key: [阿里云控制台](https://dashscope.console.aliyun.com/apiKey)"
        },
        model: {
            description: "千问模型版本",
            options: {
                plus: "平衡版 - 性价比最高 (推荐)",
                turbo: "快速版 - 响应更快,成本更低",
                max: "增强版 - 能力最强,成本较高"
            }
        },
        language: {
            description: "界面语言 / Interface Language",
            options: {
                zhCN: "简体中文",
                enUS: "English"
            }
        },
        enableContext: {
            description: "自动读取编辑器中的代码作为上下文"
        }
    }
};
