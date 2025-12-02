const axios = require('axios');
const fs = require('fs');
const path = require('path');
const KnowledgeBaseQuery = require('./knowledge_base_query'); // 引入知识库查询类

class QwenClient {
    constructor(options = {},extensionPath, language = 'zh-CN') {
        this.apiKey = options.apiKey;
        this.model = options.model;
        this.extensionPath = extensionPath;
        this.language = language;
        this.baseURL = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation';

        // 加载语言包
        this.i18n = this._loadLanguagePack();

        // 加载知识库
        //this.knowledgeBase = this._loadKnowledgeBase();
        //console.log('✅ 知识库加载完成, 长度:', this.knowledgeBase.length);

        const kbServiceUrl = options.kbServiceUrl || 'http://localhost:5000';
        const kbTimeout = options.kbTimeout || 30000;
        
        try {
            this.knowledgeBase = new KnowledgeBaseQuery({
                serviceUrl: kbServiceUrl,
                timeout: kbTimeout
            });
            console.log('✅ 知识库查询客户端初始化成功');
        } catch (error) {
            console.warn('⚠️ 知识库查询客户端初始化失败:', error.message);
            this.knowledgeBase = null;
        }

        this.ragEnabled = options.ragEnabled !== false;
        this.ragTopK = options.ragTopK || 3;

        console.log('✅ QwenClient 初始化完成');
        console.log('📚 RAG 状态:', this.ragEnabled ? '已启用' : '已禁用');
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

    _getDefaultKnowledge() {
        // 如果文件不存在,使用默认的核心物理与工程知识
        return `# 超导量子计算核心知识库 (Superconducting Quantum Computing)

## 1. 基础物理模型 (Fundamental Physics)

### Transmon 量子比特
Transmon 是目前最主流的超导量子比特类型，处于 $E_J/E_C \gg 1$ (通常 > 50) 的区间。
- **哈密顿量**: $\hat{H} \approx 4E_C(\hat{n}-n_g)^2 - E_J\cos\hat{\phi}$
- **比特频率**: $\omega_q \approx \sqrt{8E_J E_C} - E_C$
- **非简谐性 (Anharmonicity)**: $\alpha \approx -E_C$
- **适用场景**: 对电荷噪声（Charge Noise）不敏感，相干时间较长。

### 谐振腔 (Resonator)
用于读出（Readout）或作为量子总线（Bus）。
- **等效电路**: LC 振荡回路
- **频率**: $\omega_r = 1/\sqrt{LC}$
- **特性阻抗**: $Z_0 = \sqrt{L/C}$ (通常设计为 50$\Omega$)

### 色散读出 (Dispersive Readout)
在非共振条件下 ($|\Delta| = |\omega_q - \omega_r| \gg g$)，量子比特与腔的相互作用导致腔频率发生偏移。
- **色散位移 (Chi)**: $\chi \approx \frac{g^2}{\Delta} \frac{\alpha}{\Delta + \alpha}$
- **状态判别**: 通过测量微波透射信号 $S_{21}$ 的相移或幅度变化来区分 $|0\rangle$ 和 $|1\rangle$。

## 2. 关键性能指标 (Key Metrics)

- **$T_1$ (能量弛豫时间)**: 量子比特从 $|1\rangle$ 衰变到 $|0\rangle$ 的时间。主要受介质损耗（Dielectric Loss）、准粒子隧穿等影响。
- **$T_2^*$ (拉姆齐退相干时间)**: 叠加态相位信息的丢失时间。\$1/T_2^* = 1/(2T_1) + 1/T_\phi$（$T_\phi$ 为纯退相干时间）。
- **$Q$ (品质因数)**: $Q = \omega \cdot \text{Stored Energy} / \text{Power Loss}$。分为内部品质因数 $Q_{int}$ 和耦合品质因数 $Q_{ext}$。
- **门保真度 (Gate Fidelity)**: 常用 Randomized Benchmarking (RB) 方法测量。

## 3. 常用Python工具库参考
- **QuTiP**: 开放量子系统模拟 (求解主方程, 演化 dynamics)。
- **Qiskit Metal**: 超导芯片版图设计与电磁仿真接口。
- **scqubits**: 超导量子比特能谱与参数计算专用库。`;
    }

    _buildSystemPrompt() {
        const isEnglish = this.language === 'en-US';

        const roleDescription = isEnglish
            ? `You are a Senior Physicist and Engineer specializing in Superconducting Quantum Computing.

## Your Role
- Provide rigorous theoretical derivations and parameter estimations.
- Assist in experimental design (Circuit QED architecture).
- Explain physical phenomena (noise mechanisms, Hamiltonian evolution).
- Provide Python code for simulation (using QuTiP/scqubits) or data analysis.

## Core Knowledge Base
${this.knowledgeBase}`
            : `你是超导量子计算领域的资深物理学家与工程专家。

## 你的角色定位
- 提供严谨的理论推导与芯片参数估算
- 辅助实验设计（Circuit QED 架构设计）
- 解释物理现象（噪声机制、哈密顿量演化、色散读出原理）
- 提供用于模拟（QuTiP/scqubits）或数据分析的Python代码

## 核心知识库
${this.knowledgeBase}`;

        return roleDescription + (isEnglish ? `

## Important Rules - Must Follow Strictly!
1. **Scientific Rigor**: All formulas must use standard LaTeX format. Distinguish clearly between theoretical approximations (e.g., RWA) and exact solutions.
2. **Parameter Units**: Always explicitly state units (GHz, MHz, ns, $\mu$s, fF, nH). In superconducting QC, $\hbar=1$ implies frequencies are angular ($\omega$) or cyclic ($f=\omega/2\pi$). Be clear about \$2\pi$ factors.
3. **Simulation Logic**: When writing code, prefer **QuTiP** or **scqubits** standards.
4. **Physical Intuition**: Explain the "why" behind the math (e.g., why large $E_J/E_C$ suppresses charge noise).

## Common Misconceptions - Avoid These!
❌ Wrong: Confusing Coupling Strength ($g$) with Coupler Frequency.
✅ Correct: $g$ is the interaction rate (MHz); Couplers often have their own tunable frequency.

❌ Wrong: Assuming $T_2$ can be larger than \$2T_1$.
✅ Correct: Theoretical limit is $T_2 \le 2T_1$.

## Typical Analysis Workflow (Standard Template)

**Warning**: When asked to design or simulate a system, follow this logical flow:

\`\`\`python
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# 1. Define Physical Constants & Parameters
# Units: GHz, ns
h_bar = 1.0  # Working in natural units where h_bar=1 is common in QuTiP, but track 2pi!
freq_q = 5.0 * 2 * np.pi  # Qubit frequency (angular)
anharm = -0.25 * 2 * np.pi # Anharmonicity
freq_r = 6.0 * 2 * np.pi  # Resonator frequency
g_coup = 0.1 * 2 * np.pi  # Coupling strength

# 2. Construct Hamiltonian (e.g., Jaynes-Cummings or Transmon model)
# Using Duffing oscillator approximation for Transmon
a = qt.destroy(3) # Resonator operator (truncated)
q = qt.destroy(3) # Qubit operator (truncated to 3 levels to see leakage)

# H_sys = w_r * a.dag() * a + w_q * q.dag() * q + (alpha/2) * q.dag()*q.dag()*q*q + g(a.dag()*q + a*q.dag())
H = (freq_r * qt.tensor(a.dag() * a, qt.qeye(3)) +
     freq_q * qt.tensor(qt.qeye(3), q.dag() * q) +
     0.5 * anharm * qt.tensor(qt.qeye(3), q.dag() * q.dag() * q * q) +
     g_coup * (qt.tensor(a.dag(), q) + qt.tensor(a, q.dag())))

# 3. Time Evolution / Dynamics Simulation
# e.g., Rabi Oscillation or T1 decay
psi0 = qt.tensor(qt.basis(3, 0), qt.basis(3, 1)) # Resonator vacuum, Qubit excited
tlist = np.linspace(0, 50, 200)
# Collapse operators for dissipation
c_ops = [np.sqrt(1/1000.0) * qt.tensor(qt.qeye(3), q)] # Example T1=1000ns

result = qt.mesolve(H, psi0, tlist, c_ops, [])

# 4. Visualization & Analysis
plt.plot(tlist, result.expect[1]) # Plot qubit population
\`\`\`

Please follow this rigorous approach when answering.`
            : `

## 重要规则 - 必须严格遵守!
1. **科学严谨性**: 所有公式必须使用标准 LaTeX 格式。明确区分理论近似（如旋转波近似 RWA）与精确解。
2. **参数单位**: 必须明确标注单位（GHz, MHz, ns, $\mu$s, fF, nH）。在超导量子计算中，注意角频率 $\omega$ 与频率 $f$ 之间 \$2\pi$ 的区别。
3. **代码规范**: 编写模拟代码时，优先使用 **QuTiP** 或 **scqubits** 库的标准写法。
4. **物理直觉**: 在解释数学公式时，必须解释其背后的物理机制（例如：为什么高 $E_J/E_C$ 比能抑制电荷噪声）。

## 常见误区 - 避免这些错误!
❌ 错误: 混淆耦合强度 ($g$) 与 耦合器频率。
✅ 正确: $g$ 是相互作用速率（通常为 MHz 量级）；耦合器（Coupler）通常指可调频率的中间元件。

❌ 错误: 认为 $T_2$ 可以大于 \$2T_1$。
✅ 正确: 理论极限是 $T_2 \le 2T_1$。如果实验数据违背此规律，通常是拟合错误。

❌ 错误: 忽略 Transmon 的非简谐性，直接当做二能级系统处理高功率驱动。
✅ 正确: 在强驱动下必须考虑向 $|2\rangle$ 态泄漏（Leakage）的风险。

## 典型分析工作流（标准模板）

**提示**: 当被要求设计系统或模拟动力学时，请遵循以下逻辑流程：

\`\`\`python
import numpy as np
import qutip as qt
import scqubits as scq
import matplotlib.pyplot as plt

# 1. 定义物理常数与系统参数
# 所有的频率单位建议统一为 GHz (或 rad/ns)，并明确 2pi 因子
f_qubit = 5.0     # Qubit frequency in GHz
f_res = 6.0       # Resonator frequency in GHz
alpha = -0.25     # Anharmonicity in GHz
g_strength = 0.08 # Coupling strength in GHz

# 2. 构建系统模型 (哈密顿量)
# 示例：使用 scqubits 构建 Transmon 对象 (更精确，考虑了 EJ/EC)
qubit = scq.Transmon(
    EJ=20.0,
    EC=0.25,
    ng=0.0,
    ncut=30
)
# 获取能级结构
evals = qubit.eigenvals(evals_count=4)

# 或者使用 QuTiP 构建有效模型 (Jaynes-Cummings + Duffing)
# 注意：转换为角频率进行演化计算
w_q = f_qubit * 2 * np.pi
w_r = f_res * 2 * np.pi
alpha_w = alpha * 2 * np.pi
g_w = g_strength * 2 * np.pi

dim = 3 # 截断维数
a = qt.destroy(dim) # 腔算符
q = qt.destroy(dim) # 比特算符

H = (w_r * qt.tensor(a.dag() * a, qt.qeye(dim)) +
     w_q * qt.tensor(qt.qeye(dim), q.dag() * q) +
     0.5 * alpha_w * qt.tensor(qt.qeye(dim), q.dag() * q.dag() * q * q) +
     g_w * (qt.tensor(a.dag(), q) + qt.tensor(a, q.dag())))

# 3. 动力学模拟 / 演化 (Time Evolution)
# 示例：拉比振荡 (Rabi Oscillation)
psi0 = qt.tensor(qt.basis(dim, 0), qt.basis(dim, 0)) # 初始态 |0,0>
# 添加驱动项 H_drive ...

# 4. 结果可视化与分析
# 绘制布居数 (Population) 或 频谱图
\`\`\`

请始终遵循严谨的物理定义和上述分析流程来回答问题。`);
    }

    // 创建实例

    // ========================================
    // 示例 1: 基础搜索
    // ========================================
    async basicSearchExample() {
        // 检查服务是否可用
        const available = await knowledgeBase.isAvailable();
        if (!available) {
            console.log('❌ 知识库服务不可用，请先启动 Python 服务');
            return;
        }

        // 搜索
        const results = await knowledgeBase.search('量子比特如何设计');
        
        console.log(`找到 ${results.length} 条结果：`);
        results.forEach((doc, i) => {
            console.log(`\n--- 结果 ${i + 1} ---`);
            console.log(`内容: ${doc.page_content.substring(0, 100)}...`);
            console.log(`来源: ${doc.metadata?.source_file || '未知'}`);
            console.log(`距离: ${doc.distance}`);
        });
    }

    // ========================================
    // 示例 2: 带参数的搜索
    // ========================================
    async  advancedSearchExample() {
        const results = await knowledgeBase.search('谐振器频率设置', {
            k: 10,  // 返回10条结果
            filter: {
                domain: 'quantum'  // 按元数据过滤
            }
        });
        
        console.log('搜索结果:', results);
    }

    // ========================================
    // 示例 3: 搜索并格式化为上下文
    // ========================================
    async searchAndFormatExample() {
        const available = await knowledgeBase.isAvailable();
        if (!available) {
            console.log('❌ 知识库服务不可用，请先启动 Python 服务');
            return;
        }

        const context = await knowledgeBase.searchAndFormat('EDA-Q 如何创建芯片', 3);
        
        if (context) {
            console.log('格式化后的上下文：');
            console.log(context);
        } else {
            console.log('未找到相关内容');
        }
    }

    async chat(userMessage, context = {}, conversationHistory = [], onProgress = null) {
        try {
            
            const useKnowledgeBase = context.useKnowledgeBase !== false;
            const kbTopK = context.kbTopK || 3;
            let knowledgeContext = '';
            let retrievedDocs = [];

            // ========================================
            // Step 1: 从知识库检索相关内容
            // ========================================
            if (useKnowledgeBase) {
                console.log('🔍 正在检索知识库...');
                
                // 方式一：使用 search 获取原始结果
                retrievedDocs = await this.knowledgeBase.search(userMessage, { k: kbTopK });
                
                // 方式二：使用 searchAndFormat 直接获取格式化文本
                knowledgeContext = await this.knowledgeBase.searchAndFormat(userMessage, kbTopK);
                
                if (retrievedDocs.length > 0) {
                    console.log(`✅ 找到 ${retrievedDocs.length} 条相关文档`);
                } else {
                    console.log('📭 未找到相关文档');
                }
            }
            // ========================================
            // 📌 打印 knowledgeContext 内容
            // ========================================
            console.log('========== 知识库检索结果 ==========');
            console.log('📚 retrievedDocs 数量:', retrievedDocs.length);
            console.log('📚 retrievedDocs 内容:', JSON.stringify(retrievedDocs, null, 2));
            console.log('------------------------------------');
            console.log('📄 knowledgeContext 长度:', knowledgeContext ? knowledgeContext.length : 0);
            console.log('📄 knowledgeContext 内容:');
            console.log(knowledgeContext || '(空)');
            console.log('====================================');
            
            if (retrievedDocs.length > 0) {
                console.log(`✅ 找到 ${retrievedDocs.length} 条相关文档`);
            } else {
                console.log('📭 未找到相关文档');
            }
            // 构建用户提示
            let fullPrompt = '';

            if (knowledgeContext) {
                fullPrompt += knowledgeContext + '\n';
            }
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
                    timeout: 100000
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
                    timeout: 100000
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
