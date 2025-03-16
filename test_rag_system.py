#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG系统测试脚本
用于验证和演示RAG系统的各项功能
"""

import os
import sys
import argparse
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到路径，确保能够正确导入模块
sys.path.append(str(Path(__file__).parent.parent))

def test_vector_model():
    """测试向量模型"""
    from rag_system.models.vector_model import get_vector_model
    
    logger.info("开始测试向量模型...")
    
    # 初始化向量模型
    vector_model = get_vector_model()
    
    # 测试文本嵌入
    test_text = "这是一个测试文本，用于验证向量模型的嵌入功能。"
    embedding = vector_model(test_text)
    
    logger.info(f"向量维度: {embedding.shape}")
    logger.info(f"向量前几个值: {embedding[:5]}")
    
    # 测试批量嵌入
    test_texts = ["第一个测试文本", "第二个测试文本", "第三个测试文本"]
    embeddings = vector_model(test_texts)
    
    logger.info(f"批量嵌入结果维度: {embeddings.shape}")
    logger.info("向量模型测试完成！")
    
    return True

def test_llm_model(api_key):
    """测试LLM模型"""
    from rag_system.models.llm_model import get_llm_model
    
    if not api_key:
        logger.error("未提供OpenRouter API密钥，跳过LLM模型测试")
        return False
    
    logger.info("开始测试LLM模型...")
    
    # 设置环境变量
    os.environ["OPENROUTER_API_KEY"] = api_key
    
    # 初始化LLM模型
    llm_model = get_llm_model()
    
    # 测试生成
    test_prompt = "请简要介绍一下你自己。"
    response = llm_model(test_prompt)
    
    logger.info(f"LLM回复: {response}")
    logger.info("LLM模型测试完成！")
    
    return True

def test_document_processing():
    """测试文档处理"""
    from rag_system.utils.document_processor import get_document_processor
    import glob
    
    logger.info("开始测试文档处理...")
    
    # 查找markdown文件
    md_files = glob.glob("/root/autodl-fs/*.md")
    
    if not md_files:
        logger.error("未找到Markdown文件进行测试")
        return False
    
    logger.info(f"找到 {len(md_files)} 个Markdown文件")
    
    # 处理文档
    processor = get_document_processor()
    documents = processor.process_documents(md_files)
    
    logger.info(f"文档处理结果: {len(documents)} 个文档块")
    if documents:
        logger.info(f"第一个文档块: {documents[0]['text'][:100]}...")
    
    logger.info("文档处理测试完成！")
    
    return True

def test_full_rag_system(api_key):
    """测试完整的RAG系统"""
    from rag_system.rag_system import get_rag_system
    import glob
    
    if not api_key:
        logger.error("未提供OpenRouter API密钥，跳过完整RAG系统测试")
        return False
    
    logger.info("开始测试完整RAG系统...")
    
    # 设置环境变量
    os.environ["OPENROUTER_API_KEY"] = api_key
    
    # 初始化RAG系统
    rag = get_rag_system(collection_name="test_collection", api_key=api_key)
    
    # 索引文档
    md_files = glob.glob("/root/autodl-fs/*.md")
    
    if not md_files:
        logger.error("未找到Markdown文件进行测试")
        return False
    
    logger.info(f"开始索引 {len(md_files)} 个文档...")
    rag.index_documents(md_files)
    
    # 执行查询
    test_query = "GPT在量化投资中有哪些应用？"
    logger.info(f"测试查询: {test_query}")
    
    response = rag.query(test_query)
    logger.info(f"RAG系统回复:\n{response}")
    
    logger.info("完整RAG系统测试完成！")
    
    return True

def test_directory_indexing(api_key):
    """测试目录索引功能"""
    from rag_system.rag_system import get_rag_system
    
    if not api_key:
        logger.error("未提供OpenRouter API密钥，跳过目录索引测试")
        return False
    
    logger.info("开始测试目录索引功能...")
    
    # 设置环境变量
    os.environ["OPENROUTER_API_KEY"] = api_key
    
    # 初始化RAG系统
    rag = get_rag_system(collection_name="directory_test", api_key=api_key)
    
    # 索引目录
    rag.index_directory("/root/autodl-fs", pattern="*.md")
    
    # 执行查询
    test_query = "这些文档主要讨论了什么内容？"
    logger.info(f"测试查询: {test_query}")
    
    response = rag.query(test_query)
    logger.info(f"RAG系统回复:\n{response}")
    
    logger.info("目录索引测试完成！")
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG系统测试脚本")
    parser.add_argument("--api_key", type=str, default="", help="OpenRouter API密钥")
    parser.add_argument("--test_vector", action="store_true", help="测试向量模型")
    parser.add_argument("--test_llm", action="store_true", help="测试LLM模型")
    parser.add_argument("--test_doc", action="store_true", help="测试文档处理")
    parser.add_argument("--test_rag", action="store_true", help="测试完整RAG系统")
    parser.add_argument("--test_dir_index", action="store_true", help="测试目录索引")
    parser.add_argument("--test_all", action="store_true", help="测试所有功能")
    
    args = parser.parse_args()
    
    # 如果没有指定任何测试，默认测试所有功能
    if not (args.test_vector or args.test_llm or args.test_doc or args.test_rag or args.test_dir_index):
        args.test_all = True
    
    # 提示用户输入API密钥（如果未提供）
    api_key = args.api_key
    if not api_key and (args.test_llm or args.test_rag or args.test_dir_index or args.test_all):
        api_key = input("请输入OpenRouter API密钥: ").strip()
    
    # 执行测试
    test_results = {}
    
    if args.test_vector or args.test_all:
        test_results["向量模型"] = test_vector_model()
    
    if args.test_llm or args.test_all:
        test_results["LLM模型"] = test_llm_model(api_key)
    
    if args.test_doc or args.test_all:
        test_results["文档处理"] = test_document_processing()
    
    if args.test_rag or args.test_all:
        test_results["完整RAG系统"] = test_full_rag_system(api_key)
    
    if args.test_dir_index or args.test_all:
        test_results["目录索引"] = test_directory_indexing(api_key)
    
    # 打印测试结果汇总
    logger.info("\n测试结果汇总:")
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{test_name}: {status}")

if __name__ == "__main__":
    main()
