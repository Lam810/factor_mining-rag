#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置文件，包含RAG系统所需的所有配置参数
"""

import os
from pathlib import Path

# 基础路径配置
BASE_DIR = Path("/root/autodl-fs/rag_system")
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
UTILS_DIR = BASE_DIR / "utils"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# 确保所有目录存在
for dir_path in [DATA_DIR, MODELS_DIR, UTILS_DIR, VECTOR_DB_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# OpenRouter配置
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")  # 需要设置环境变量或在此处填入API密钥
OPENROUTER_MODEL = "deepseek/deepseek-r1-zero:free"

# 向量模型配置
VECTOR_MODEL = "intfloat/e5-mistral-7b-instruct"
VECTOR_MODEL_TASK = "embed"

# 并发处理配置
MAX_WORKERS = 4  # 并发处理的工作进程数

# 文档处理配置
CHUNK_SIZE = 1000  # 文档分块大小
CHUNK_OVERLAP = 200  # 块之间的重叠大小

# 检索配置
TOP_K_RETRIEVAL = 5  # 检索时返回的相似文档数量
