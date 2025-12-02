from flask import Flask, request, jsonify
import numpy as np
from typing import List
import os

# 复用你原来的类
from faiss_h1 import OptimizedEmbedder, FaissIndexHandler, SQLiteHandler

app = Flask(__name__)

# 全局初始化（只加载一次）
print("正在初始化服务...")
embedder = OptimizedEmbedder(model_path=r"D:\5j\project\rag\bge-large-zh-v1.5")
faiss_handler = FaissIndexHandler(index_path="./my_kb/index.bin", embedding_dim=1024)
sql_handler = SQLiteHandler(db_path="./my_kb/data.db")
print("✅ 服务初始化完成")


@app.route('/search', methods=['POST'])
def search():
    """
    知识库搜索接口
    请求体: {"query": "搜索内容", "k": 5, "filter": {"domain": "xxx"}}
    """
    try:
        data = request.json
        query = data.get('query', '')
        k = data.get('k', 5)
        filter_metadata = data.get('filter', None)
        
        if not query:
            return jsonify({"error": "query 不能为空"}), 400
        
        # 1. 生成查询向量
        query_vector = embedder.embed_query(query)
        
        # 2. FAISS 搜索 (L2 距离)
        distances, doc_ids = faiss_handler.search(query_vector, k=k*2)
        
        # 3. 过滤无效 ID
        valid_ids = [int(i) for i in doc_ids if i != -1]
        
        # 4. 从 SQLite 获取文档内容
        documents = sql_handler.get_documents_by_ids(valid_ids)
        
        # 5. 元数据过滤
        results = []
        for i, doc in enumerate(documents):
            if filter_metadata:
                match = True
                for key, val in filter_metadata.items():
                    if doc['metadata'].get(key) != val:
                        match = False
                        break
                if not match:
                    continue
            
            # 添加距离信息
            doc['distance'] = float(distances[i]) if i < len(distances) else None
            results.append(doc)
            
            if len(results) >= k:
                break
        
        return jsonify({
            "success": True,
            "query": query,
            "count": len(results),
            "results": results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
