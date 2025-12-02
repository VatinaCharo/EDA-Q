module.exports = {
    // UI界面文本
    ui: {
        header: {
            title: "🔬 超导量子计算知识助手",
            subtitle: "基于 LLM + RAG | 超导量子计算领域专家"
        },
        welcome: {
            icon: "🌟",
            title: "欢迎使用超导量子计算知识助手！",
            description: "我可以回答超导量子计算相关问题、解释量子物理概念、介绍前沿研究进展，基于专业知识库为您提供准确可靠的答案",
            examplesTitle: "💡 试试这些问题：",
            examples: [
                {
                    icon: "⚛️",
                    text: "什么是量子纠缠？它有哪些应用？",
                    question: "什么是量子纠缠？它有哪些实际应用？"
                },
                {
                    icon: "🔢",
                    text: "Shor算法的原理是什么？",
                    question: "请详细解释Shor算法的工作原理及其对密码学的影响"
                },
                {
                    icon: "🛡️",
                    text: "量子纠错是如何实现的？",
                    question: "量子纠错的基本原理是什么？常用的纠错码有哪些？"
                },
                {
                    icon: "🔮",
                    text: "超导量子比特有哪些类型？",
                    question: "超导量子比特有哪些主要类型？各有什么特点？"
                }
            ]
        },
        quickActions: [
            { icon: "📖", text: "基础入门", question: "请介绍超导量子计算的基本概念和原理" },
            { icon: "🔍", text: "深入解释", question: "请更详细地解释这个概念" },
            { icon: "📝", text: "总结归纳", question: "请总结一下以上内容的关键要点" },
            { icon: "🗑️", text: "清空对话", action: "clear" }
        ],
        input: {
            placeholder: "输入您的超导量子计算问题... (Shift+Enter 换行，Enter 发送)",
            sendButton: "提问",
            sendingButton: "思考中..."
        },
        codeBlock: {
            copyButton: "📋 复制",
            saveButton: "💾 保存笔记",
            copied: "✓ 已复制"
        },
        thinking: {
            analyzing: "🤔 正在分析您的问题",
            searching: "🔍 正在检索知识库",
            generating: "💡 正在生成专业回答"
        },
        messages: {
            errorPrefix: "❌ ",
            noteSaved: "✅ 笔记已保存",
            noteSaveFailed: "❌ 保存笔记失败",
            contentCopied: "✅ 内容已复制到剪贴板",
            historyCleared: "对话历史已清空",
            clearConfirm: "确定要清空对话历史吗？"
        },
        notifications: {
            activated: "🔬 超导量子计算知识助手已就绪！",
            openAssistant: "打开助手"
        },
        sources: {
            title: "📚 参考来源",
            showMore: "显示更多来源",
            showLess: "收起来源"
        },
        feedback: {
            helpful: "👍 有帮助",
            notHelpful: "👎 需改进",
            thanks: "感谢您的反馈！"
        }
    },

    // 错误消息
    errors: {
        noApiKey: "请先在设置中配置 API Key\n\n设置路径：文件 → 首选项 → 设置 → 搜索 \"量子助手\"",
        apiKeyInvalid: "API Key 无效或已过期\n\n请检查：\n1. API Key 是否正确\n2. API Key 是否已激活\n3. 账户是否有足够余额",
        rateLimitExceeded: "请求过于频繁\n\n建议：\n1. 稍等片刻后重试\n2. 考虑升级 API 套餐",
        badRequest: "请求参数错误",
        timeout: "请求超时\n\n可能原因：\n1. 网络连接不稳定\n2. API 服务响应较慢\n\n请重试",
        networkError: "无法连接到 API 服务\n\n请检查：\n1. 网络连接是否正常\n2. 是否需要配置代理",
        unknownError: "未知错误",
        apiFormatError: "API 响应格式错误",
        knowledgeBaseError: "知识库加载失败\n\n请检查 my_kb 目录是否完整",
        embeddingError: "文本向量化处理出错",
        modelLoadError: "BGE 模型加载失败\n\n请检查 my_kb/bge-large-zh-v1.5 目录",
        retryOrContact: "请重试或联系技术支持"
    },

    // 配置项描述
    config: {
        apiKey: {
            description: "阿里云通义千问 API Key（必填）",
            markdownDescription: "获取 API Key：[阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/apiKey)"
        },
        model: {
            description: "通义千问模型版本",
            options: {
                plus: "均衡版 - 性价比最高（推荐）",
                turbo: "快速版 - 响应更快，成本更低",
                max: "增强版 - 能力最强，成本较高"
            }
        },
        language: {
            description: "界面语言",
            options: {
                zhCN: "简体中文",
                enUS: "English"
            }
        },
        enableRAG: {
            description: "启用 RAG 检索增强生成（基于本地知识库提供更准确的回答）"
        },
        topK: {
            description: "检索相关文档数量（1-10，数值越大参考内容越多）"
        },
        similarityThreshold: {
            description: "相似度阈值（0-1，越高匹配越严格）"
        }
    }
};