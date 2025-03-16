#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
向量模型封装，使用VLLM驱动本地向量模型
"""

import os
import numpy as np
from typing import List, Union, Dict, Any
from vllm import LLM
import torch
from tqdm import tqdm

from ..config import VECTOR_MODEL, VECTOR_MODEL_TASK, MAX_WORKERS

class VectorEmbedder:
    """使用VLLM驱动的向量模型封装"""
    
    def __init__(self, model: str = VECTOR_MODEL, task: str = VECTOR_MODEL_TASK):
        """
        初始化向量模型
        
        Args:
            model: 向量模型名称或路径
            task: 模型任务类型，通常是'embed'
        """
        self.model_name = model
        self.task = task
        
        # 检查GPU是否可用
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 初始化VLLM模型
        self.llm = LLM(model=model, task=task)
        print(f"成功加载向量模型: {model}")
    
    def embed(self, text: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        生成文本的嵌入向量
        
        Args:
            text: 单个文本或文本列表
            batch_size: 批处理大小
            
        Returns:
            嵌入向量或嵌入向量列表
        """
        if isinstance(text, str):
            result = self.llm.embed(text)
            return result[0]  # VLLM返回的结果是元组形式
        
        # 批量处理文本列表
        results = []
        total_batches = (len(text) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(text), batch_size), total=total_batches, desc="生成嵌入向量"):
            batch_texts = text[i:i+batch_size]
            batch_results = []
            
            for single_text in batch_texts:
                # 对每个文本单独处理
                result = self.llm.embed(single_text)
                batch_results.append(result[0])
            
            results.extend(batch_results)
        
        return np.array(results)
    
    def embed_documents(self, documents: List[Dict[str, Any]], text_key: str = "text") -> List[Dict[str, Any]]:
        """
        为文档列表生成嵌入向量
        
        Args:
            documents: 文档列表，每个文档是一个字典
            text_key: 文档字典中文本内容的键名
            
        Returns:
            添加了嵌入向量的文档列表
        """
        texts = [doc[text_key] for doc in documents]
        embeddings = self.embed(texts)
        
        for i, doc in enumerate(documents):
            doc["embedding"] = embeddings[i]
        
        return documents
    
    def __call__(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        简化调用接口
        
        Args:
            text: 文本或文本列表
            
        Returns:
            嵌入向量或嵌入向量列表
        """
        return self.embed(text)


# 单例模式，避免重复加载大模型
_vector_model_instance = None

def get_vector_model() -> VectorEmbedder:
    """获取向量模型实例（单例模式）"""
    global _vector_model_instance
    if _vector_model_instance is None:
        _vector_model_instance = VectorEmbedder()
    return _vector_model_instance
