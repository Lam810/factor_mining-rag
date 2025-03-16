#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文档处理模块，用于加载和分割文档
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import re
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import logging

from ..config import CHUNK_SIZE, CHUNK_OVERLAP, MAX_WORKERS

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """文档处理器，处理各种文档的加载和分割"""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        初始化文档处理器
        
        Args:
            chunk_size: 分块大小
            chunk_overlap: 块之间的重叠大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_markdown(self, file_path: Union[str, Path]) -> str:
        """
        加载Markdown文档
        
        Args:
            file_path: Markdown文件路径
            
        Returns:
            文档内容
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content
    
    def split_text(self, text: str) -> List[str]:
        """
        将文本分割成块
        
        Args:
            text: 要分割的文本
            
        Returns:
            分割后的文本块列表
        """
        # 确保文本是字符串
        if not isinstance(text, str):
            raise ValueError(f"预期文本是字符串，但收到 {type(text)}")
        
        # 如果文本很短，直接返回
        if len(text) < self.chunk_size:
            return [text]
        
        # 分割文本
        chunks = []
        start = 0
        
        while start < len(text):
            # 找到chunk_size之后的第一个段落分隔符
            end = min(start + self.chunk_size, len(text))
            
            # 如果end不是文本的末尾，尝试找到一个更好的断点
            if end < len(text):
                # 尝试在段落或句子边界上断开
                paragraph_end = text.find('\n\n', end - self.chunk_size // 2, end + self.chunk_size // 2)
                sentence_end = text.find('. ', end - self.chunk_size // 4, end + self.chunk_size // 4)
                
                if paragraph_end != -1:
                    end = paragraph_end + 2  # 包含段落分隔符
                elif sentence_end != -1:
                    end = sentence_end + 2  # 包含句号和空格
            
            # 添加块
            chunks.append(text[start:end])
            
            # 移动起点，考虑重叠
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_document(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        处理单个文档
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            处理后的文档块列表
        """
        try:
            # 加载文档
            content = self.load_markdown(file_path)
            
            # 提取元数据
            file_name = Path(file_path).name
            
            # 分割文档
            chunks = self.split_text(content)
            
            # 创建文档块
            documents = []
            for i, chunk in enumerate(chunks):
                doc = {
                    "text": chunk,
                    "metadata": {
                        "source": str(file_path),
                        "file_name": file_name,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                }
                documents.append(doc)
            
            return documents
        
        except Exception as e:
            logger.error(f"处理文档 {file_path} 时出错: {e}")
            return []
    
    def process_documents(self, file_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        并行处理多个文档
        
        Args:
            file_paths: 文档文件路径列表
            
        Returns:
            处理后的所有文档块列表
        """
        # 如果文件数量少，直接串行处理
        if len(file_paths) <= 1:
            all_documents = []
            for file_path in tqdm(file_paths, desc="处理文档"):
                documents = self.process_document(file_path)
                all_documents.extend(documents)
            return all_documents
        
        # 并行处理多个文档
        num_workers = min(MAX_WORKERS, len(file_paths), multiprocessing.cpu_count())
        
        logger.info(f"启动 {num_workers} 个工作进程处理 {len(file_paths)} 个文档")
        
        with Pool(num_workers) as pool:
            # 使用tqdm显示进度
            results = list(tqdm(
                pool.imap(self.process_document, file_paths),
                total=len(file_paths),
                desc="并行处理文档"
            ))
        
        # 合并结果
        all_documents = []
        for documents in results:
            all_documents.extend(documents)
        
        return all_documents


# 创建单例
_document_processor_instance = None

def get_document_processor() -> DocumentProcessor:
    """获取文档处理器实例（单例模式）"""
    global _document_processor_instance
    if _document_processor_instance is None:
        _document_processor_instance = DocumentProcessor()
    return _document_processor_instance
