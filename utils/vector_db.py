#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
向量数据库模块，使用ChromaDB存储和检索文档向量
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import logging
import numpy as np

from ..config import VECTOR_DB_DIR, TOP_K_RETRIEVAL

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorDatabase:
    """使用ChromaDB的向量数据库"""
    
    def __init__(self, collection_name: str = "documents", persist_directory: Union[str, Path] = VECTOR_DB_DIR):
        """
        初始化向量数据库
        
        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        
        # 确保目录存在
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"加载已存在的集合: {collection_name}")
        except ValueError:
            self.collection = self.client.create_collection(name=collection_name)
            logger.info(f"创建新集合: {collection_name}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        添加文档到数据库
        
        Args:
            documents: 包含文本和嵌入向量的文档列表
        """
        if not documents:
            logger.warning("没有文档可添加")
            return
        
        # 准备批量添加的数据
        ids = []
        embeddings = []
        metadatas = []
        contents = []
        
        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}_{hash(doc['text'])}"
            
            # 添加到批量列表
            ids.append(doc_id)
            embeddings.append(doc.get('embedding', None))  # 如果没有嵌入，将由ChromaDB生成
            metadatas.append(doc['metadata'])
            contents.append(doc['text'])
        
        # 批量添加到ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings if all(emb is not None for emb in embeddings) else None,
            metadatas=metadatas,
            documents=contents
        )
        
        logger.info(f"成功添加 {len(documents)} 个文档到向量数据库")
    
    def search(self, query_embedding: np.ndarray, top_k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """
        搜索相似文档
        
        Args:
            query_embedding: 查询的嵌入向量
            top_k: 返回的最相似文档数量
            
        Returns:
            相似文档列表
        """
        # 执行查询
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # 处理结果
        documents = []
        for i in range(len(results['documents'][0])):
            doc = {
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            }
            documents.append(doc)
        
        return documents
    
    def get_collection_count(self) -> int:
        """
        获取集合中的文档数量
        
        Returns:
            文档数量
        """
        return self.collection.count()
    
    def delete_collection(self) -> None:
        """删除集合"""
        self.client.delete_collection(self.collection_name)
        logger.info(f"删除集合: {self.collection_name}")


# 创建单例
_vector_db_instance = None

def get_vector_db(collection_name: str = "documents") -> VectorDatabase:
    """获取向量数据库实例（单例模式）"""
    global _vector_db_instance
    if _vector_db_instance is None:
        _vector_db_instance = VectorDatabase(collection_name=collection_name)
    return _vector_db_instance
