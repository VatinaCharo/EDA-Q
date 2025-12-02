import os
import torch
from typing import List, Optional

class OptimizedEmbedder:
    """针对 RTX 5080 优化的 Embedding 类"""
    
    def __init__(
        self, 
        model_path: str = r"D:\5j\project\rag\bge-large-zh-v1.5",
        device: Optional[str] = None,
        batch_size: int = 128,  # 5080 显存大，可以开大
        use_fp16: bool = True,  # 半精度加速
    ):
        self.model_path = model_path
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        
        # 自动检测设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self._print_gpu_info()
        self._load_model()
    
    def _print_gpu_info(self):
        """打印 GPU 信息"""
        if torch.cuda.is_available():
            print("=" * 50)
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"PyTorch 版本: {torch.__version__}")
            print(f"使用设备: {self.device}")
            print(f"半精度模式: {self.use_fp16}")
            print("=" * 50)
        else:
            print("⚠️ 未检测到 GPU，将使用 CPU（速度较慢）")
    
    def _load_model(self):
        """加载模型"""
        from sentence_transformers import SentenceTransformer
        
        if not os.path.exists(self.model_path):
            raise ValueError(f"模型路径不存在: {self.model_path}")
        
        print(f"正在加载模型: {self.model_path}")
        
        self.model = SentenceTransformer(
            self.model_path,
            device=self.device,
            trust_remote_code=True,
        )
        
        # 半精度优化（显著提升速度，几乎不影响效果）
        if self.use_fp16 and self.device == 'cuda':
            self.model = self.model.half()
            print("✅ 已启用 FP16 半精度加速")
        
        print("✅ 模型加载完成")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        if not texts:
            return []
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()


# 集成到 LangChain
from langchain_core.embeddings import Embeddings

class FastGPUEmbeddings(Embeddings):
    """LangChain 兼容的 GPU 加速 Embeddings"""
    
    def __init__(
        self,
        model_path: str = r"D:\5j\project\rag\bge-large-zh-v1.5",
        batch_size: int = 128,
        use_fp16: bool = True,
    ):
        self.embedder = OptimizedEmbedder(
            model_path=model_path,
            batch_size=batch_size,
            use_fp16=use_fp16,
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        return self.embedder.embed_query(text)

import os
import sqlite3
import json
import numpy as np
import faiss
import pickle
from typing import List, Dict, Tuple
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# ==========================================
# 1. SQLite 数据库管理类 (存文本和元数据)
# ==========================================
class SQLiteHandler:
    def __init__(self, db_path="knowledge_base.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # 创建表：ID(自增), 内容, 元数据(JSON字符串), 向量(Blob, 可选备份)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                page_content TEXT,
                metadata TEXT,
                source_file TEXT  -- 单独提取出来方便快速过滤
            )
        ''')
        # 创建索引加速查询
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON documents (source_file)')
        conn.commit()
        conn.close()

    def insert_documents(self, docs_data: List[Dict]) -> List[int]:
        """
        批量插入文档，返回生成的 ID 列表
        :param docs_data: list of dict, [{"page_content": "...", "metadata": {...}}]
        :return: ids list
        """
        if not docs_data:
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        inserted_ids = []
        
        try:
            # 方法1: 使用事务 + 逐条插入（推荐，兼顾性能和准确性）
            conn.execute("BEGIN TRANSACTION")
            
            for doc in docs_data:
                content = doc.get("page_content", "")
                meta = doc.get("metadata", {})
                source = meta.get("source_file", "unknown")
                meta_json = json.dumps(meta, ensure_ascii=False)
                
                cursor.execute(
                    'INSERT INTO documents (page_content, metadata, source_file) VALUES (?, ?, ?)',
                    (content, meta_json, source)
                )
                inserted_ids.append(cursor.lastrowid)
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
        
        return inserted_ids

    def get_documents_by_ids(self, ids: List[int]) -> List[Dict]:
        """根据 ID 列表取回完整的文档内容"""
        if not ids:
            return []
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 动态构建 SQL: SELECT * FROM documents WHERE id IN (1, 2, 3...)
        placeholders = ','.join(['?'] * len(ids))
        query = f'SELECT id, page_content, metadata FROM documents WHERE id IN ({placeholders})'
        
        cursor.execute(query, ids)
        rows = cursor.fetchall()
        conn.close()
        
        # 将结果转回字典，并保持原来的 ID 顺序
        # 数据库返回的顺序可能不一致，这里做一个映射
        id_map = {row[0]: {"page_content": row[1], "metadata": json.loads(row[2])} for row in rows}
        
        results = []
        for doc_id in ids:
            if doc_id in id_map:
                results.append(id_map[doc_id])
        return results
    
# ==========================================
# 2. FAISS 向量管理类 (只存向量和ID映射)
# ==========================================
class FaissIndexHandler:
    def __init__(self, index_path="faiss_index.bin", embedding_dim=1024):
        """
        :param embedding_dim: 向量维度 (BGE-large通常是1024, OpenAI是1536)
        """
        self.index_path = index_path
        self.dim = embedding_dim
        
        if os.path.exists(index_path):
            print(f"加载现有 FAISS 索引: {index_path}")
            self.index = faiss.read_index(index_path)
        else:
            print("新建 FAISS 索引")
            # 使用 IndexIDMap，这样我们可以手动指定 ID，让它与 SQLite 的 ID 对应
            self.index = faiss.IndexIDMap2(faiss.IndexFlatL2(self.dim))

    def add_vectors(self, vectors: List[List[float]], ids: List[int]):
        """
        添加向量及其对应的 ID
        """
        if len(vectors) == 0:
            return
            
        np_vectors = np.array(vectors).astype('float32')
        np_ids = np.array(ids).astype('int64')
        
        # 添加到索引
        self.index.add_with_ids(np_vectors, np_ids)

    def search(self, query_vector: List[float], k=3 , threshold: float = 0.75):
        """
        搜索最近邻
        :return: (distances, ids)
        """
        np_query = np.array([query_vector]).astype('float32')
        distances, ids = self.index.search(np_query, k)

        # 获取单条查询的原始结果（因为 search 支持批量，这里取第一条）
        raw_dists = distances[0]
        raw_ids = ids[0]
        
        # 3. 核心优化逻辑：使用布尔掩码（Boolean Mask）进行过滤
        # 假设是欧氏距离（L2）：距离越小越好，所以保留 distance <= threshold 的结果
        # 如果所有结果都 > threshold，这里会返回空数组，实现了"最近的也不要"的需求
        valid_mask = raw_dists <= threshold
        
        # 应用掩码，只保留符合条件的数据
        filtered_dists = raw_dists[valid_mask]
        filtered_ids = raw_ids[valid_mask]
        
        return filtered_dists, filtered_ids
        #return distances[0], ids[0]

    def save(self):
        faiss.write_index(self.index, self.index_path)

class KnowledgeBaseManager:
    def __init__(self, db_path="./data/kb.db", faiss_path="./data/index.bin", model_name="bge"):
        # 确保目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.sql_handler = SQLiteHandler(db_path)
        
        # 加载 Embedding 模型
        self.embedding_model = self._load_embedding_model(model_name)
        
        # 获取模型维度 (hack: 跑一次空向量看维度，或者手动指定)
        sample_emb = self.embedding_model.embed_query("test")
        emb_dim = len(sample_emb)
        
        self.faiss_handler = FaissIndexHandler(faiss_path, embedding_dim=emb_dim)

    def _load_embedding_model(self, model_name):
        if model_name == "bge":
            # 【重点修改这里】
            # 1. 这里填你刚才下载的文件夹的绝对路径
            # 注意 Windows 路径要用双斜杠 \\ 或者在引号前加 r
             return FastGPUEmbeddings(
                model_path=r"D:\5j\project\rag\bge-large-zh-v1.5",
                batch_size=128,
                use_fp16=True,
            )
        elif model_name == "openai":
            return OpenAIEmbeddings()
        return None

    def add_data(self, chunks_list: List[Dict]):
        """
        核心流程：
        1. 文本 -> SQLite -> 获取 IDs
        2. 文本 -> Embedding -> 向量
        3. 向量 + IDs -> FAISS
        """
        if not chunks_list:
            return

        print("1. 正在存入 SQLite...")
        ids = self.sql_handler.insert_documents(chunks_list)
        
        print("2. 正在生成向量 (Embeddings)...")
        texts = [c["page_content"] for c in chunks_list]
        embeddings = self.embedding_model.embed_documents(texts)
        
        print(f"3. 正在更新 FAISS 索引 (插入 {len(embeddings)} 条)...")
        self.faiss_handler.add_vectors(embeddings, ids)
        
        # 持久化保存
        self.faiss_handler.save()
        print("入库完成！")

    def search(self, query: str, k=3, filter_metadata: Dict = None):
        """
        核心搜索流程：
        1. Query -> Embedding
        2. FAISS -> 搜索得到 TopK IDs
        3. IDs -> SQLite -> 查出具体文本
        4. (可选) Python层进行 Metadata 过滤
        """
        # 1. 向量化
        query_vec = self.embedding_model.embed_query(query)
        
        # 2. 向量搜索
        distances, doc_ids = self.faiss_handler.search(query_vec, k=k*2) # 多取一点，防止过滤后不够
        
        # 过滤掉无效 ID (-1 代表没找到)
        valid_ids = [int(i) for i in doc_ids if i != -1]
        
        # 3. 回表查询详细信息
        retrieved_docs = self.sql_handler.get_documents_by_ids(valid_ids)
        
        # 4. 后置过滤 (Post-Filtering)
        # 因为 FAISS 很难做复杂的元数据过滤，我们先取出向量相似的，再用代码筛一遍
        final_results = []
        for doc in retrieved_docs:
            if filter_metadata:
                match = True
                for key, val in filter_metadata.items():
                    if doc['metadata'].get(key) != val:
                        match = False
                        break
                if not match:
                    continue
            final_results.append(doc)
            
            if len(final_results) >= k:
                break
                
        return final_results

# ==========================================
# 4. 使用示例
# ==========================================

import json
import os

# ... (上面的 SQLiteHandler, FaissIndexHandler, KnowledgeBaseManager 类保持不变) ...

def load_chunks_from_json(file_path):
    """
    从 JSON 文件读取数据
    预期格式: [{"page_content": "...", "metadata": {...}}, ...]
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件路径不存在 -> {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            print(f"成功从文件加载 {len(data)} 条 Chunk 数据")
            return data
        else:
            print("错误: JSON 文件根节点不是列表 (List)，无法处理。")
            return []
    except json.JSONDecodeError:
        print("错误: 文件内容不是有效的 JSON 格式。")
        return []
    except Exception as e:
        print(f"读取文件时发生未知错误: {e}")
        return []
    
if __name__ == "__main__":
    # ---------------------------------------------------------
    # 配置部分
    # ---------------------------------------------------------
    json_file_path = "D:\\5j\\project\\rag\\chunks_without_martinis.json" # 修改为你真实的 JSON 文件路径
    
    # 确保你的 Embedding 模型环境变量已设置，或者在初始化时指定
    # os.environ["bge"] = "/path/to/your/model" 

    # 1. 从文件读取 Data Batch
    data_batch = load_chunks_from_json(json_file_path)

    # 只有成功读取到数据才进行后续操作
    if data_batch:
        # 2. 初始化管理器
        # 如果是第一次运行，会自动创建数据库和索引文件
        # 如果不是第一次，会自动加载旧的并在其基础上追加
        kb = KnowledgeBaseManager(
            db_path="./my_kb/data.db", 
            faiss_path="./my_kb/index.bin", 
            model_name="bge" # 如果没有本地模型，这里改成 "openai" 并设置 api key
        )
        
        # 3. 入库 (批量写入)
        # 建议：如果数据量特别大(几万条)，这里可以写个循环分批传入 add_data
        #kb.add_data(data_batch)
        
        # 4. 搜索测试
        print("\n--- 开始搜索测试 ---")
        search_query = "详细解释一下shor算法"
        
        # 示例：搜索并进行 Metadata 过滤
        results = kb.search(
            search_query, 
            k=10,
            filter_metadata={"domain": "超导量子芯片"} # 如果你的 JSON 里没有这个字段，请注释掉这行
        )
        
        if not results:
            print(f"未找到关于 '{search_query}' 的相关内容。")

        for i, res in enumerate(results):
            print(f"\n>>> 搜索结果 {i+1}:")
            print(f"来源文件: {res['metadata'].get('source_file', 'Unknown')}")
            print(f"页码: {res['metadata'].get('page_label', 'Unknown')}")
            # 打印内容摘要（去除换行符方便显示）
            content_preview = res['page_content'][:100].replace('\n', ' ')
            print(f"内容摘要: {content_preview}...")
    else:
        print("程序终止：未获取到有效数据。")