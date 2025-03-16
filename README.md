# RAG系统使用说明

这是一个基于本地向量模型和OpenRouter驱动的深度学习模型构建的检索增强生成（RAG）系统。该系统可以对Markdown文档进行索引，并根据用户的查询提供基于文档内容的智能回答。

## 系统架构

RAG系统由以下主要组件构成：

1. **向量模型**：使用VLLM驱动的`intfloat/e5-mistral-7b-instruct`模型，用于生成文本的嵌入向量。
2. **LLM模型**：使用OpenRouter API驱动的`deepseek/deepseek-r1-zero:free`模型，用于生成回复。
3. **文档处理器**：负责加载和分割Markdown文档。
4. **向量数据库**：使用ChromaDB存储和检索文档向量。
5. **RAG系统**：整合以上组件，提供完整的检索增强生成功能。

```
┌─────────────┐        ┌────────────────┐        ┌───────────────┐
│ Markdown文档 │───────>│ 文档处理器     │───────>│ 向量模型      │
└─────────────┘        └────────────────┘        └───────┬───────┘
                                                         │
                                                         ▼
                       ┌────────────────┐        ┌───────────────┐
                       │ 用户查询       │───────>│ 向量模型      │
                       └────────────────┘        └───────┬───────┘
                                                         │
                                                         ▼
┌─────────────┐        ┌────────────────┐        ┌───────────────┐
│ 生成回复    │<───────│ LLM模型        │<───────│ 向量数据库    │
└─────────────┘        └────────────────┘        └───────────────┘
```

## 功能特点

- **处理Markdown文档**：自动加载和分割Markdown文件，支持批量处理。
- **并发处理**：支持并行处理大量文档，高效生成嵌入向量。
- **语义检索**：基于向量相似度的精准文档检索。
- **上下文感知**：将检索到的相关文档作为上下文提供给LLM模型。
- **持久化存储**：向量数据库支持持久化，索引结果可以重复使用。

## 安装指南

### 环境要求

- Python 3.8+
- CUDA支持（推荐用于加速向量模型）

### 安装步骤

1. 确保已经安装所有必要的依赖库：

```bash
source /root/autodl-fs/rag_venv/bin/activate
pip install openrouter vllm langchain langchain-openrouter langchain_community sentence-transformers chromadb tqdm multiprocessing-on-dill
```

2. 配置OpenRouter API密钥

   在使用系统前，需要设置OpenRouter API密钥。可以通过以下方式：

   - 在`config.py`中直接设置`OPENROUTER_API_KEY`
   - 设置环境变量`OPENROUTER_API_KEY`
   - 在实例化RAG系统时提供API密钥

## 使用方法

### 基本使用

```python
# 导入RAG系统
from rag_system.rag_system import get_rag_system

# 初始化RAG系统（提供API密钥）
rag = get_rag_system(api_key="your_openrouter_api_key")

# 索引一个或多个文档
rag.index_documents(["/path/to/document1.md", "/path/to/document2.md"])

# 或者索引整个目录
rag.index_directory("/path/to/documents/", pattern="*.md")

# 执行查询
response = rag.query("你的问题")
print(response)
```

### 测试脚本

系统提供了一个测试脚本，可以验证各组件功能：

```bash
cd /root/autodl-fs
source rag_venv/bin/activate
python -m rag_system.test_rag_system --api_key="your_openrouter_api_key"
```

可以使用以下参数测试特定功能：

- `--test_vector`：测试向量模型
- `--test_llm`：测试LLM模型
- `--test_doc`：测试文档处理
- `--test_rag`：测试完整RAG系统
- `--test_dir_index`：测试目录索引
- `--test_all`：测试所有功能（默认）

## 配置选项

系统参数可以在`config.py`中配置：

- **向量模型**：可以更改模型名称和任务类型
- **块大小**：文档分割的块大小和重叠量
- **检索数量**：每次查询返回的相似文档数量
- **并发数**：文档处理的并发数量

## 常见问题

**Q: 是否支持中文文档？**  
A: 是的，系统完全支持中文文档的处理和查询。

**Q: 如何处理大量文档？**  
A: 系统采用并行处理方式，可以高效处理大量文档。通过调整`MAX_WORKERS`参数可以控制并发度。

**Q: 向量数据库的存储位置在哪里？**  
A: 向量数据库默认存储在`/root/autodl-fs/rag_system/data/vector_db`目录下。

**Q: 如何清除已索引的数据？**  
A: 可以使用以下代码清除集合：
```python
from rag_system.utils.vector_db import get_vector_db
db = get_vector_db()
db.delete_collection()
```

## 系统扩展

RAG系统设计为模块化结构，可以方便地扩展和定制：

- **替换向量模型**：修改`config.py`中的`VECTOR_MODEL`参数
- **更换LLM模型**：修改`config.py`中的`OPENROUTER_MODEL`参数
- **添加文档类型**：扩展`document_processor.py`中的处理方法
- **优化检索逻辑**：修改`vector_db.py`中的搜索方法

## 开发者指南

系统采用单例模式设计，各组件可以独立使用：

```python
# 使用向量模型
from rag_system.models.vector_model import get_vector_model
vector_model = get_vector_model()
embedding = vector_model("测试文本")

# 使用文档处理器
from rag_system.utils.document_processor import get_document_processor
processor = get_document_processor()
documents = processor.process_document("/path/to/document.md")

# 使用向量数据库
from rag_system.utils.vector_db import get_vector_db
db = get_vector_db()
db.add_documents(documents_with_embeddings)
```

---

希望这个RAG系统能够满足您的需求！如有任何问题或建议，请随时提出。
