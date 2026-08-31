<div align="center">

# factor-rag

**切在表格之间，而不是切穿表格。**

面向视觉丰富文档的结构感知 Markdown 切分——
让一张 48 行的财务表格不再被拦腰截断、丢掉表头之后送进模型。

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-17%20passing-brightgreen.svg)](tests/)
[![Zero heavy deps](https://img.shields.io/badge/chunker%20deps-0-orange.svg)](#安装)

[English](README.md) · **简体中文**

</div>

---

## 问题

把 PDF 丢给版面解析器——MinerU、Marker、Nougat、PP-StructureV2——吐出来的东西
主要是**结构**：管道表格、图片引用、公式、标题层级。

然后一个按字符数切分的 chunker 每 1000 字符切一刀，结构就没了。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/structure-map-dark.svg">
  <img alt="同一文档上的切分边界：字符窗口 vs 结构感知" src="assets/structure-map-light.svg">
</picture>

图中每一条红色横杠，都是一行**与表头失散**的表格数据。它同时是不可检索的——没有任何词汇
信号，只剩 `| 0.31 | 0.08 |`——也是不可用的，因为没有模型能读懂一张没有表头的表。在自带的
样本语料上，这是 **94 行中的 60 行**，几乎全部来自文档里最长的那两张表。

## 解法

`chunk_markdown()` 先把文档解析成语义块，再进行装箱：

| 保证 | 含义 |
|---|---|
| **表格永不丢表头** | 超长表格按行切分，表头行在每一段中重复出现 |
| **代码围栏保持配对** | 代码块或公式块绝不会从中间被切断 |
| **图片不与标题分离** | 图片引用与其题注保持在同一 chunk 内 |
| **每个 chunk 知道自己的章节** | 标题路径同时写入正文与元数据 |
| **保证终止** | 原先的字符窗口切分器会让游标倒退，可能死循环 |
| **ID 确定** | 基于内容与来源的 SHA-1，而非带随机盐的 `hash()`——重复索引变成更新而不是重复写入 |

## 用数字说话，而不是用形容词

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/fidelity-dark.svg">
  <img alt="切分后的结构保真度：baseline 与结构感知对比" src="assets/fidelity-light.svg">
</picture>

| 指标 | 字符窗口 | 结构感知 |
|---|---|---|
| 表格完整率 | 78% | **100%** |
| 保留表头的表格行占比 | 79% | **100%** |
| 能说出自身章节的 chunk 占比 | 81% | **100%** |
| 与题注相邻的图片占比 | 100% | **100%** |

在 3 篇样本语料上、`chunk_size=1000` 时的均值。从干净 clone 即可复现：

```bash
python benchmarks/run_benchmark.py
```

不需要模型、不需要 API key、不需要 GPU——语料就在 [`samples/`](samples/) 里，
且所有指标都只从 chunk 文本计算，因此任何切分策略都能在同一标准下被打分。

## 安装

切分器**没有重依赖**——不需要 torch、vLLM 或 LangChain：

```bash
pip install -e .
```

需要完整检索链路时：

```bash
pip install -e ".[rag]"
```

## 使用

多数人只需要切分本身：

```python
from factor_rag import chunk_markdown

chunks = chunk_markdown(open("report.md", encoding="utf-8").read(),
                        source="report.md", chunk_size=1000)

for c in chunks:
    print(c.metadata["heading_path"], c.metadata["has_table"], len(c.text))
```

chunk 的元数据全是标量，可以直接塞进 Chroma、FAISS 或 Qdrant，无需再做展平。

给任意切分策略打分：

```python
from factor_rag import score_chunks, naive_chunk

report = score_chunks(markdown, naive_chunk(markdown, 1000, 200))
print(report.table_integrity, report.row_header_coverage)
```

完整链路：

```python
from factor_rag.rag_system import RAGSystem

rag = RAGSystem()
rag.index_directory("docs/")
print(rag.query("盈亏平衡的交易成本是多少？"))
```

需先在环境变量中设置 `OPENROUTER_API_KEY`。所有配置均由环境变量驱动
（`FACTOR_RAG_HOME`、`FACTOR_RAG_CHUNK_SIZE`、`FACTOR_RAG_TOP_K` 等），
详见 [`factor_rag/config.py`](factor_rag/config.py)。

## 0.2 版改了什么

这一版与其说是加功能，不如说是修复。0.1 版的包**根本无法被导入**：
`utils/document_processor.py` 写的是 `from ..config import ...`，而顶层包名含连字符，
不是合法的模块名；`models/llm_model.py` 导入的 `langchain_openrouter` 在 PyPI 上并不存在。
除了新增结构感知切分器，0.2 还修掉了不终止的切分逻辑、带进程随机盐的文档 ID、
接不住 Chroma `NotFoundError` 的 `except ValueError`、把 vLLM 请求对象当作向量直接使用的
嵌入读取，以及导入时就会在文件系统上创建目录的硬编码 `/root/autodl-fs` 路径。

## 引用

```bibtex
@software{lin_factor_rag,
  author  = {Lin, Zeteng},
  title   = {factor-rag: Structure-Aware Chunking for Retrieval over Visually Rich Documents},
  year    = {2026},
  url     = {https://github.com/Lam810/factor_mining-rag}
}
```

## 许可

MIT，见 [LICENSE](LICENSE)。

由 **林泽腾**（Zeteng Lin）开发，香港科技大学（广州）信息枢纽数据科学与分析学域博士研究生。
[lam810.github.io](https://lam810.github.io/)
