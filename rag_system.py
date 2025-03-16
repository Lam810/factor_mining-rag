#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG系统主类，整合所有组件
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
from glob import glob
import time

from .models.vector_model import get_vector_model
from .models.llm_model import get_llm_model
from .utils.document_processor import get_document_processor
from .utils.vector_db import get_vector_db
from .config import OPENROUTER_API_KEY

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystem:
    """RAG系统类，整合向量模型、LLM、文档处理和向量数据库"""
    
    def __init__(self, collection_name: str = "documents", api_key: Optional[str] = None):
        """
        初始化RAG系统
        
        Args:
            collection_name: 向量数据库集合名称
            api_key: OpenRouter API密钥（可选，如不提供将使用config中的配置）
        """
        # 设置API密钥
        if api_key:
            os.environ["OPENROUTER_API_KEY"] = api_key
        elif not OPENROUTER_API_KEY:
            logger.warning("未提供OpenRouter API密钥，请确认已在环境变量中设置或在实例化时提供")
        
        # 初始化组件
        logger.info("正在初始化RAG系统组件...")
        
        # 向量模型可能需要更长时间加载
        try:
            start_time = time.time()
            self.vector_model = get_vector_model()
            logger.info(f"向量模型加载完成，耗时: {time.time() - start_time:.2f}秒")
        except Exception as e:
            logger.error(f"向量模型加载失败: {e}")
            raise
        
        try:
            self.document_processor = get_document_processor()
            self.vector_db = get_vector_db(collection_name=collection_name)
            logger.info("文档处理器和向量数据库初始化完成")
        except Exception as e:
            logger.error(f"初始化文档处理器或向量数据库失败: {e}")
            raise
        
        # 只有在需要时才初始化LLM模型
        self._llm_model = None
        
        logger.info("RAG系统初始化完成")
    
    @property
    def llm_model(self):
        """懒加载LLM模型"""
        if self._llm_model is None:
            try:
                start_time = time.time()
                self._llm_model = get_llm_model()
                logger.info(f"LLM模型加载完成，耗时: {time.time() - start_time:.2f}秒")
            except Exception as e:
                logger.error(f"LLM模型加载失败: {e}")
                raise
        return self._llm_model
    
    def index_documents(self, file_paths: List[Union[str, Path]]) -> int:
        """
        索引文档到向量数据库
        
        Args:
            file_paths: 文档文件路径列表
            
        Returns:
            成功索引的文档数量
        """
        logger.info(f"开始处理 {len(file_paths)} 个文档...")
        
        # 处理文档
        documents = self.document_processor.process_documents(file_paths)
        logger.info(f"文档处理完成，生成了 {len(documents)} 个文档块")
        
        # 生成嵌入向量
        logger.info("开始生成嵌入向量...")
        start_time = time.time()
        documents_with_embeddings = self.vector_model.embed_documents(documents)
        logger.info(f"嵌入向量生成完成，耗时: {time.time() - start_time:.2f}秒")
        
        # 存储到向量数据库
        logger.info("开始存储文档到向量数据库...")
        self.vector_db.add_documents(documents_with_embeddings)
        
        # 获取数据库中的文档数量
        doc_count = self.vector_db.get_collection_count()
        logger.info(f"向量数据库中现有 {doc_count} 个文档")
        
        return len(documents)
    
    def index_directory(self, directory_path: Union[str, Path], pattern: str = "*.md") -> int:
        """
        索引目录中的所有文档
        
        Args:
            directory_path: 目录路径
            pattern: 文件名匹配模式
            
        Returns:
            成功索引的文档数量
        """
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"目录不存在或不是一个有效的目录: {directory_path}")
        
        # 获取所有匹配的文件
        file_pattern = os.path.join(str(dir_path), pattern)
        file_paths = glob(file_pattern, recursive=True)
        
        if not file_paths:
            logger.warning(f"在目录 {directory_path} 中未找到匹配 {pattern} 的文件")
            return 0
        
        logger.info(f"在目录 {directory_path} 中找到 {len(file_paths)} 个匹配的文件")
        
        # 索引文档
        return self.index_documents(file_paths)
    
    def query(self, query: str, top_k: int = 5) -> str:
        """
        执行查询
        
        Args:
            query: 查询文本
            top_k: 检索的文档数量
            
        Returns:
            生成的回复
        """
        logger.info(f"收到查询: {query}")
        
        # 生成查询嵌入向量
        query_embedding = self.vector_model(query)
        
        # 检索相似文档
        logger.info(f"从向量数据库检索 {top_k} 个相似文档...")
        results = self.vector_db.search(query_embedding, top_k=top_k)
        
        if not results:
            logger.warning("未找到相关文档")
            return "抱歉，我无法找到与您问题相关的信息。"
        
        # 提取文档文本
        contexts = [doc["text"] for doc in results]
        logger.info(f"找到 {len(contexts)} 个相关文档块")
        
        # 生成回复
        logger.info("正在生成回复...")
        system_prompt = "你是一个专业助手，基于提供的文档回答用户问题。只使用提供的文档内容回答，不要编造信息。"
        response = self.llm_model.rag_generate(query, contexts, system_prompt)
        
        return response

# 创建单例
_rag_system_instance = None

def get_rag_system(collection_name: str = "documents", api_key: Optional[str] = None) -> RAGSystem:
    """获取RAG系统实例（单例模式）"""
    global _rag_system_instance
    if _rag_system_instance is None:
        _rag_system_instance = RAGSystem(collection_name=collection_name, api_key=api_key)
    return _rag_system_instance
