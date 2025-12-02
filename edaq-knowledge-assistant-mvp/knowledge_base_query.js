const axios = require('axios');

/**
 * 知识库查询客户端
 * 调用 Python 微服务进行向量检索
 */
class KnowledgeBaseQuery {
    /**
     * @param {Object} options
     * @param {string} options.serviceUrl - Python 服务地址
     * @param {number} options.timeout - 请求超时时间(ms)
     */
    constructor(options = {}) {
        this.serviceUrl = options.serviceUrl || 'http://localhost:5000';
        this.timeout = options.timeout || 30000;
        
        console.log(`📚 知识库查询服务: ${this.serviceUrl}`);
    }

    /**
     * 检查服务是否可用
     * @returns {Promise<boolean>}
     */
    async isAvailable() {
        try {
            const response = await axios.get(
                `${this.serviceUrl}/health`,
                { timeout: 5000 }
            );
            return response.data.status === 'ok';
        } catch (error) {
            console.warn('⚠️ 知识库服务不可用:', error.message);
            return false;
        }
    }

    /**
     * 搜索知识库
     * @param {string} query - 用户查询
     * @param {Object} options - 搜索选项
     * @param {number} options.k - 返回结果数量
     * @param {Object} options.filter - 元数据过滤条件
     * @returns {Promise<Array>} - 搜索结果
     */
    async search(query, options = {}) {
        const { k = 5, filter = null } = options;

        try {
            const response = await axios.post(
                `${this.serviceUrl}/search`,
                {
                    query: query,
                    k: k,
                    filter: filter
                },
                { timeout: this.timeout }
            );

            if (response.data.success) {
                return response.data.results;
            } else {
                console.error('搜索失败:', response.data.error);
                return [];
            }
        } catch (error) {
            console.error('❌ 知识库查询出错:', error.message);
            return [];
        }
    }

    /**
     * 搜索并格式化为上下文字符串
     * @param {string} query - 用户查询
     * @param {number} k - 返回数量
     * @returns {Promise<string>} - 格式化的上下文
     */
    async searchAndFormat(query, k = 3) {
        const results = await this.search(query, { k });

        if (results.length === 0) {
            return '';
        }

        let context = '【检索到的相关知识】\n\n';

        results.forEach((doc, index) => {
            const source = doc.metadata?.source_file || '未知来源';
            const page = doc.metadata?.page_label || '';
            
            context += `--- 参考资料 ${index + 1} ---\n`;
            context += `来源: ${source}${page ? ` (第${page}页)` : ''}\n`;
            context += `${doc.page_content}\n\n`;
        });

        return context;
    }
}

module.exports = KnowledgeBaseQuery;
