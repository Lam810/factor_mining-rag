#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM模型封装，使用OpenRouter驱动的deepseek/deepseek-r1-zero:free模型
"""

import os
from typing import Dict, List, Any, Optional
import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openrouter import OpenRouter
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage

from ..config import OPENROUTER_API_KEY, OPENROUTER_MODEL

class LLMModel:
    """OpenRouter驱动的LLM模型封装"""
    
    def __init__(self, api_key: str = OPENROUTER_API_KEY, model_name: str = OPENROUTER_MODEL):
        """
        初始化LLM模型
        
        Args:
            api_key: OpenRouter API密钥
            model_name: 模型名称
        """
        if not api_key:
            raise ValueError("OpenRouter API密钥不能为空，请在config.py中配置或设置环境变量OPENROUTER_API_KEY")
        
        self.api_key = api_key
        self.model_name = model_name
        
        # 初始化OpenRouter模型
        self.llm = OpenRouter(
            api_key=api_key,
            model=model_name,
            temperature=0.7,
            max_tokens=4096,
        )
        
        print(f"成功初始化OpenRouter LLM模型: {model_name}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        生成文本
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示（可选）
            
        Returns:
            生成的回复
        """
        messages = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        messages.append(HumanMessage(content=prompt))
        
        response = self.llm.invoke(messages)
        return response.content
    
    def rag_generate(self, query: str, context: List[str], system_prompt: Optional[str] = None) -> str:
        """
        基于检索上下文生成回复
        
        Args:
            query: 用户查询
            context: 检索到的上下文列表
            system_prompt: 系统提示（可选）
            
        Returns:
            生成的回复
        """
        # 合并所有上下文
        context_text = "\n\n".join([f"文档 {i+1}:\n{ctx}" for i, ctx in enumerate(context)])
        
        # 构建RAG提示
        rag_prompt = f"""
根据以下提供的参考文档，回答用户的问题。如果无法从参考文档中找到答案，请诚实地说明无法从提供的文档中找到相关信息。

参考文档:
{context_text}

用户问题:
{query}
"""
        
        # 生成回复
        return self.generate(rag_prompt, system_prompt)
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """
        简化调用接口
        
        Args:
            prompt: 用户提示
            **kwargs: 其他参数
            
        Returns:
            生成的回复
        """
        return self.generate(prompt, **kwargs)


# 单例模式，避免重复初始化
_llm_model_instance = None

def get_llm_model() -> LLMModel:
    """获取LLM模型实例（单例模式）"""
    global _llm_model_instance
    if _llm_model_instance is None:
        _llm_model_instance = LLMModel()
    return _llm_model_instance
