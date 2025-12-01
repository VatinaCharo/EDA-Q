const axios = require('axios');
const fs = require('fs');
const path = require('path');

class QwenClient {
    constructor(apiKey, model = 'qwen-plus', extensionPath, language = 'zh-CN') {
        this.apiKey = apiKey;
        this.model = model;
        this.extensionPath = extensionPath;
        this.language = language;
        this.baseURL = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation';

        // 加载语言包
        this.i18n = this._loadLanguagePack();

        // 加载知识库
        this.knowledgeBase = this._loadKnowledgeBase();
        console.log('✅ 知识库加载完成, 长度:', this.knowledgeBase.length);
    }

    _loadLanguagePack() {
        try {
            const langFile = this.language === 'en-US' ? 'en-US.js' : 'zh-CN.js';
            const langPath = path.join(this.extensionPath, 'locales', langFile);

            if (fs.existsSync(langPath)) {
                // 清除缓存以确保重新加载
                delete require.cache[require.resolve(langPath)];
                return require(langPath);
            } else {
                console.warn('⚠️ 语言包文件不存在:', langPath);
                return require(path.join(this.extensionPath, 'locales', 'zh-CN.js'));
            }
        } catch (error) {
            console.error('❌ 加载语言包失败:', error);
            return require(path.join(this.extensionPath, 'locales', 'zh-CN.js'));
        }
    }

    _loadKnowledgeBase() {
        try {
            const manualFile = this.language === 'en-US' ? 'user_manual_en.txt' : 'user_manual_zh.txt';
            const manualPath = path.join(this.extensionPath, 'knowledge', manualFile);

            if (fs.existsSync(manualPath)) {
                const content = fs.readFileSync(manualPath, 'utf-8');
                console.log('📚 知识库文件路径:', manualPath);
                return content;
            } else {
                console.warn('⚠️ 知识库文件不存在:', manualPath);
                return this._getDefaultKnowledge();
            }
        } catch (error) {
            console.error('❌ 加载知识库失败:', error);
            return this._getDefaultKnowledge();
        }
    }

    _getDefaultKnowledge() {
        // 如果文件不存在,使用默认的核心知识
        return `# EDA-Q 核心API参考

## Design类主要方法

### generate_topology() - 生成拓扑
参数:
- qubits_num: 量子比特数量
- topo_row: 行数(可选)
- topo_col: 列数(可选)

示例:
design.generate_topology(qubits_num=16)

### generate_qubits() - 生成量子比特
参数:
- topology: 是否基于拓扑
- qubits_type: 类型(Transmon, Xmon等)
- dist: 间距

示例:
design.generate_qubits(topology=True, qubits_type="Transmon", dist=2000)

### 完整流程示例
from api.design import Design
design = Design()
design.generate_topology(qubits_num=16)
design.topology.generate_full_edges()
design.generate_qubits(topology=True, qubits_type="Transmon", dist=2000)
design.generate_chip(qubits=True, chip_name="chip0")
design.gds.save_gds("output.gds")`;
    }

    _buildSystemPrompt() {
        const isEnglish = this.language === 'en-US';

        const roleDescription = isEnglish
            ? `You are a professional AI assistant for EDA-Q quantum chip design tool.

## Your Role
- Focus on helping users design quantum chips with EDA-Q
- Provide accurate, executable Python code
- Explain EDA-Q APIs and concepts
- Debug and optimize user code

## Core Knowledge Base
${this.knowledgeBase}`
            : `你是 EDA-Q 量子芯片设计工具的专业AI助手。

## 你的角色定位
- 专注于帮助用户使用EDA-Q进行量子芯片设计
- 提供准确、可执行的Python代码
- 解释EDA-Q的API和概念
- 调试和优化用户代码

## 核心知识库
${this.knowledgeBase}`;

        return roleDescription + (isEnglish ? `

## Important Rules - Must Follow Strictly!
1. **Follow Examples Strictly**: When generating code, follow the format in knowledge base examples exactly, don't create parameters
2. **Parameters Must Be Correct**: Parameter names, types, and order must match knowledge base examples
3. **Code Format**: Code must be wrapped in \`\`\`python
4. **Clear Comments**: Use ${isEnglish ? 'English' : 'Chinese'} comments for key steps
5. **Error Handling**: Point out issues first, then provide solutions

## Common Error Examples - Avoid These!
❌ Wrong: design.generate_readout_lines(rdls_type="ReadoutCavity", chip_name="chip0", qubits=True)
✅ Correct: design.generate_readout_lines(qubits=True, rdls_type="ReadoutCavity", chip_name="chip0")

❌ Wrong: Creating non-existent parameters
✅ Correct: Only use parameters explicitly listed in knowledge base

## Typical Workflow (Standard Template - Order is CRITICAL!)

**Warning**: The following step order has been validated and absolutely cannot be changed! Otherwise it will cause serious errors!

**Default Scale**: If user doesn't specify qubit count, generate 64-qubit chip (8x8 grid) by default

\`\`\`python
from api.design import Design

# 1. Create Design object
design = Design()

# 2. Generate topology structure (define qubit grid layout)
# Default 64 qubits: 8 rows x 8 columns grid topology
design.generate_topology(qubits_num=64)
design.topology.generate_full_edges()

# 3. Generate qubits (place qubit components on layout)
design.generate_qubits(topology=True, qubits_type="Transmon", dist=2000, chip_name="chip0")

# 4. Generate coupling lines (connect adjacent qubits)
# Note: Must be BEFORE generating chip boundary!
design.generate_coupling_lines(topology=True, qubits=True, cpls_type="CouplerBase", chip="chip0")

# 5. Generate chip boundary
# Note: Must be AFTER coupling lines!
design.generate_chip(qubits=True, dist=4000, chip_name="chip0")

# 6. [CRITICAL STEP - Cannot be skipped!] Generate readout cavities
# Parameter order must be: qubits, rdls_type, chip_name
design.generate_readout_lines(qubits=True, rdls_type="ReadoutCavity", chip_name="chip0")

# 7. Copy chip layer (for multi-layer routing design)
# Note: Must be AFTER generating readout cavities!
design.gds.chips.copy_chip(old_chip_name="chip0", new_chip_name="chip1")

# 8. Auto-routing (generate control lines and readout lines)
design.routing(method="Flipchip_routing", chip_name="chip1")

# 9. Display and save results
# Recommended: Use GDS viewer to display design (automatically opens KLayout etc.)
design.gds.show_gds()

# Other optional display methods (commented out):
# design.gds.show_svg()  # Display in browser as SVG
# design.gds.save_gds("quantum_chip_64qubits.gds")  # Save as GDS file
\`\`\`

**Critical Order Rules**:
1. Coupling lines (Step 4) MUST be BEFORE chip boundary (Step 5)
2. Chip boundary (Step 5) MUST be BEFORE readout cavities (Step 6)
3. Readout cavities (Step 6) MUST be BEFORE copy chip (Step 7)
4. Copy chip (Step 7) MUST be BEFORE routing (Step 8)

**Common Errors**:
❌ Wrong Order: chip → copy → coupling → readout → routing
✅ Correct Order: qubits → coupling → chip → readout → copy → routing

Please always follow the above rules and template when answering questions.`
            : `

## 重要规则 - 必须严格遵守!
1. **严格遵循示例**: 生成代码时必须完全按照知识库中的示例格式,不要自创参数
2. **参数必须正确**: 每个API调用的参数名称、类型、顺序必须与知识库示例一致
3. **代码格式**: 代码必须用\`\`\`python包裹
4. **注释清晰**: 关键步骤用中文注释说明
5. **错误处理**: 如发现用户代码有错,先指出问题再给解决方案

## 常见错误示例 - 避免这些错误!
❌ 错误: design.generate_readout_lines(rdls_type="ReadoutCavity", chip_name="chip0", qubits=True)
✅ 正确: design.generate_readout_lines(qubits=True, rdls_type="ReadoutCavity", chip_name="chip0")

❌ 错误: 自创不存在的参数
✅ 正确: 只使用知识库中明确列出的参数

## 典型工作流程（标准模板 - 步骤顺序极其重要!）

**警告**: 以下步骤顺序经过验证，绝对不能改变!否则会导致严重错误!

**默认规模**: 如果用户没有明确指定比特数，默认生成64比特芯片（8x8网格）

\`\`\`python
from api.design import Design

# 1. 创建Design对象
design = Design()

# 2. 生成拓扑结构（定义量子比特网格布局）
# 默认64比特：8行8列的网格拓扑
design.generate_topology(qubits_num=64)
design.topology.generate_full_edges()

# 3. 生成量子比特（在版图上放置量子比特元件）
design.generate_qubits(topology=True, qubits_type="Transmon", dist=2000, chip_name="chip0")

# 4. 生成耦合线（连接相邻量子比特）
# 注意：必须在生成chip边界之前!
design.generate_coupling_lines(topology=True, qubits=True, cpls_type="CouplerBase", chip="chip0")

# 5. 生成芯片边界
# 注意：必须在耦合线之后!
design.generate_chip(qubits=True, dist=4000, chip_name="chip0")

# 6. 【关键步骤 - 不能省略!】生成读取谐振腔
# 参数顺序必须是: qubits, rdls_type, chip_name
design.generate_readout_lines(qubits=True, rdls_type="ReadoutCavity", chip_name="chip0")

# 7. 复制芯片层（用于多层布线设计）
# 注意：必须在生成读取腔之后!
design.gds.chips.copy_chip(old_chip_name="chip0", new_chip_name="chip1")

# 8. 自动布线（生成控制线和读取线）
design.routing(method="Flipchip_routing", chip_name="chip1")

# 9. 显示和保存结果
# 推荐：使用GDS查看器展示设计（会自动打开KLayout等工具）
design.gds.show_gds()

# 其他可选展示方式（已注释）：
# design.gds.show_svg()  # 在浏览器中显示SVG格式
# design.gds.save_gds("quantum_chip_64qubits.gds")  # 保存为GDS文件
\`\`\`

**关键顺序规则**:
1. 耦合线(步骤4) 必须在 芯片边界(步骤5) 之前
2. 芯片边界(步骤5) 必须在 读取腔(步骤6) 之前
3. 读取腔(步骤6) 必须在 复制芯片(步骤7) 之前
4. 复制芯片(步骤7) 必须在 布线(步骤8) 之前

**常见错误**:
❌ 错误顺序: chip → copy → coupling → readout → routing
✅ 正确顺序: qubits → coupling → chip → readout → copy → routing

请始终遵循以上规则和模板回答问题。`);
    }

    async chat(userMessage, context = {}, conversationHistory = [], onProgress = null) {
        try {
            // 构建用户提示
            let fullPrompt = '';

            // 添加代码上下文
            if (context.currentCode && context.currentCode.trim()) {
                fullPrompt += `## 当前打开的代码文件\n`;
                fullPrompt += `文件: ${context.fileName}\n`;
                fullPrompt += `\`\`\`python\n${context.currentCode}\n\`\`\`\n\n`;
            }

            fullPrompt += `## 用户问题\n${userMessage}`;

            // 构建对话历史(保留最近5轮)
            const recentHistory = conversationHistory.slice(-10);
            const messages = [
                {
                    role: 'system',
                    content: this._buildSystemPrompt()
                },
                ...recentHistory,
                {
                    role: 'user',
                    content: fullPrompt
                }
            ];

            console.log('🚀 发送请求到千问API...');
            console.log('📝 消息数量:', messages.length);

            // 如果有进度回调，使用流式输出
            if (onProgress) {
                return await this._chatWithStream(messages, onProgress);
            }

            // 否则使用普通模式
            const response = await axios.post(
                this.baseURL,
                {
                    model: this.model,
                    input: {
                        messages: messages
                    },
                    parameters: {
                        temperature: 0.7,
                        top_p: 0.8,
                        max_tokens: 2000,
                        result_format: 'message'
                    }
                },
                {
                    headers: {
                        'Authorization': `Bearer ${this.apiKey}`,
                        'Content-Type': 'application/json',
                        'X-DashScope-SSE': 'disable'
                    },
                    timeout: 60000
                }
            );

            console.log('✅ 收到API响应');

            if (!response.data || !response.data.output) {
                throw new Error(this.i18n.errors.apiFormatError);
            }

            const assistantMessage = response.data.output.choices[0].message.content;
            const codeMatch = assistantMessage.match(/```python\n([\s\S]*?)\n```/);
            const code = codeMatch ? codeMatch[1] : null;

            if (response.data.usage) {
                const usage = response.data.usage;
                console.log('📊 Token使用:',
                    `输入=${usage.input_tokens}`,
                    `输出=${usage.output_tokens}`,
                    `总计=${usage.total_tokens}`
                );
            }

            return {
                text: assistantMessage,
                code: code
            };

        } catch (error) {
            console.error('❌ 千问API调用失败:', error.response?.data || error.message);
            throw this._handleError(error);
        }
    }

    async _chatWithStream(messages, onProgress) {
        try {
            const response = await axios.post(
                this.baseURL,
                {
                    model: this.model,
                    input: { messages: messages },
                    parameters: {
                        temperature: 0.7,
                        top_p: 0.8,
                        max_tokens: 2000,
                        incremental_output: true
                    }
                },
                {
                    headers: {
                        'Authorization': `Bearer ${this.apiKey}`,
                        'Content-Type': 'application/json',
                        'Accept': 'text/event-stream',
                        'X-DashScope-SSE': 'enable'
                    },
                    responseType: 'stream',
                    timeout: 60000
                }
            );

            let fullText = '';
            let buffer = '';

            return new Promise((resolve, reject) => {
                response.data.on('data', (chunk) => {
                    buffer += chunk.toString();
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';

                    for (const line of lines) {
                        if (line.startsWith('data:')) {
                            const data = line.slice(5).trim();
                            if (data === '[DONE]') continue;

                            try {
                                const parsed = JSON.parse(data);
                                if (parsed.output && parsed.output.choices && parsed.output.choices[0]) {
                                    const delta = parsed.output.choices[0].message.content;
                                    if (delta) {
                                        fullText += delta;
                                        onProgress(delta);
                                    }
                                }
                            } catch (e) {
                                console.warn('解析SSE数据失败:', e);
                            }
                        }
                    }
                });

                response.data.on('end', () => {
                    console.log('✅ 流式输出完成');
                    const codeMatch = fullText.match(/```python\n([\s\S]*?)\n```/);
                    const code = codeMatch ? codeMatch[1] : null;
                    resolve({ text: fullText, code: code });
                });

                response.data.on('error', (error) => {
                    console.error('❌ 流式输出错误:', error);
                    reject(error);
                });
            });

        } catch (error) {
            console.error('❌ 流式API调用失败:', error.response?.data || error.message);
            throw this._handleError(error);
        }
    }

    _handleError(error) {
        const errors = this.i18n.errors;

        if (error.response) {
            const status = error.response.status;
            const errorData = error.response.data;

            if (status === 401 || status === 403) {
                return new Error(errors.apiKeyInvalid);
            } else if (status === 429) {
                return new Error(errors.rateLimitExceeded);
            } else if (status === 400) {
                const errMsg = errorData?.message || errors.badRequest;
                return new Error(`${errors.badRequest}: ${errMsg}`);
            } else {
                return new Error(`API ${errors.unknownError} (${status}): ${errorData?.message || errors.unknownError}`);
            }
        } else if (error.code === 'ECONNABORTED') {
            return new Error(errors.timeout);
        } else if (error.code === 'ENOTFOUND' || error.code === 'ECONNREFUSED') {
            return new Error(errors.networkError);
        } else {
            return new Error(`${errors.unknownError}: ${error.message}\n\n${errors.retryOrContact}`);
        }
    }
}

module.exports = QwenClient;
