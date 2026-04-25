# RAG Service

一个可直接调用的本地 RAG 服务组件，不包含 API 层，适合先把知识库能力搭起来，再接 FastAPI、Flask 或其他服务框架。

当前实现包含 4 段能力：

1. 文档读取与切片
2. DashScope 向量化 + FAISS 向量检索
3. BM25 关键词检索
4. Cross-Encoder 精排

## 功能概览

- 支持 `.txt`、`.md`、`.pdf` 文档入库
- 使用 `RecursiveCharacterTextSplitter` 做文本切片
- 使用 `DashScopeEmbeddings(model="text-embedding-v4")` 生成向量
- 使用 `FAISS` 做向量检索
- 使用 `BM25Plus` 做关键词检索
- 使用 `Cross-Encoder` 做重排序精排
- 索引和元数据持久化到本地磁盘

## 目录说明

- `data/faiss.index`：FAISS 向量索引文件
- `data/metadata.json`：切片后的文本和元数据
- `rag_service.py`：核心实现

## 依赖

项目使用 `uv` 管理依赖，核心依赖包括：

- `faiss-cpu`
- `pypdf`
- `jieba`
- `langchain-community`
- `langchain-text-splitters`
- `rank-bm25`
- `dashscope`
- `sentence-transformers`

安装依赖：

```bash
uv sync
```

## 环境变量

向量模型使用 DashScope，所以运行前需要设置：

```bash
export DASHSCOPE_API_KEY="你的 DashScope Key"
```

如果你后面直接调用 `search()` 并启用精排，第一次还会下载 Cross-Encoder 模型，首轮会慢一些。

## 快速开始

```python
from rag_service import RAGService

rag = RAGService(data_dir="data")

rag.add_documents([
    "./docs/a.txt",
    "./docs/b.pdf",
])

results = rag.search("请帮我查找退款规则", top_k=3)
for item in results:
    print(item["score"], item["source"])
    print(item["content"])
```

## 检索流程

默认 `search()` 的执行链路是：

1. 向量召回
2. BM25 关键词召回
3. 两路结果做加权融合
4. 取前 `candidate_top_k` 个候选
5. 用 Cross-Encoder 做精排
6. 返回最终 `top_k` 结果

如果你不想精排，可以传 `use_rerank=False`。

## 返回结果格式

大多数检索函数都会返回 `list[dict]`，每个元素结构类似：

```python
{
    "score": 0.91,
    "vector_score": 0.88,
    "bm25_score": 1.0,
    "hybrid_score": 0.916,
    "rerank_score": 3.42,
    "content": "命中的文本片段",
    "source": "./docs/a.txt",
    "metadata": {"chunk_index": 0},
    "retrieval_mode": "hybrid_rerank",
}
```

字段说明：

- `score`：当前阶段最终排序分数
- `vector_score`：向量召回分数
- `bm25_score`：BM25 分数
- `hybrid_score`：混合召回分数
- `rerank_score`：Cross-Encoder 精排分数；未精排时为 `None`
- `content`：文本片段内容
- `source`：来源文件路径
- `metadata`：当前只包含 `chunk_index`
- `retrieval_mode`：当前结果来自哪种策略

## 类初始化

### `RAGService(...)`

签名：

```python
RAGService(
    data_dir: str = "data",
    model_name: str = "text-embedding-v4",
    embeddings: Any | None = None,
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
    reranker: Any | None = None,
    reranker_device: str | None = None,
    reranker_batch_size: int = 16,
    default_use_rerank: bool = True,
    default_candidate_top_k: int = 20,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
)
```

常用参数：

- `data_dir`：索引和元数据保存目录
- `model_name`：DashScope embedding 模型名
- `chunk_size`：默认切片大小
- `chunk_overlap`：默认切片重叠大小
- `reranker_model_name`：Cross-Encoder 模型名
- `default_use_rerank`：`search()` 默认是否启用精排
- `default_candidate_top_k`：默认先召回多少候选再精排

进阶参数：

- `embeddings`：可传入自定义 embedding 对象，便于测试或替换模型
- `reranker`：可传入自定义 reranker 对象，便于测试或替换精排器
- `reranker_device`：指定精排模型运行设备，如 `"cpu"`、`"cuda"`
- `reranker_batch_size`：Cross-Encoder 批量推理大小

示例：

```python
from rag_service import RAGService

rag = RAGService(
    data_dir="data",
    chunk_size=800,
    chunk_overlap=150,
    default_use_rerank=True,
    default_candidate_top_k=15,
)
```

## 可用函数

### `add_documents(file_paths, chunk_size=None, chunk_overlap=None)`

作用：

- 读取文档
- 文本切片
- 写入 FAISS
- 重建 BM25 索引
- 持久化到本地

参数：

- `file_paths`：文件路径列表
- `chunk_size`：本次入库覆盖默认切片大小
- `chunk_overlap`：本次入库覆盖默认重叠大小

返回值：

```python
{
    "added_chunks": 12,
    "sources": ["./docs/a.txt", "./docs/b.pdf"]
}
```

示例：

```python
rag.add_documents([
    "./docs/商品说明.txt",
    "./docs/退款政策.pdf",
])

rag.add_documents(
    ["./docs/客服FAQ.md"],
    chunk_size=1000,
    chunk_overlap=200,
)
```

注意：

- `chunk_overlap` 必须小于 `chunk_size`
- 支持格式只有 `.txt`、`.md`、`.pdf`

### `search_by_vector(query, top_k=3)`

作用：

- 只走向量召回
- 不使用 BM25
- 不做 Cross-Encoder 精排

适合场景：

- 想单独观察语义召回效果
- 调试 embedding 召回质量

示例：

```python
results = rag.search_by_vector("退款规则", top_k=5)
```

### `search_by_bm25(query, top_k=3)`

作用：

- 只走关键词检索
- 不使用向量召回
- 不做精排

适合场景：

- 查询词比较明确
- 想验证关键词命中情况

示例：

```python
results = rag.search_by_bm25("七天无理由 退款", top_k=5)
```

### `search_by_hybrid(query, top_k=10, vector_weight=0.7, bm25_weight=0.3)`

作用：

- 同时使用向量召回和 BM25
- 对两路分数做加权融合
- 不做精排

参数：

- `query`：查询文本
- `top_k`：返回候选数量
- `vector_weight`：向量召回权重
- `bm25_weight`：BM25 权重

说明：

- 两个权重都必须大于等于 `0`
- 两个权重不能同时为 `0`

示例：

```python
results = rag.search_by_hybrid(
    "退款规则",
    top_k=10,
    vector_weight=0.6,
    bm25_weight=0.4,
)
```

### `rerank(query, candidates, top_k=None)`

作用：

- 对已有候选结果做 Cross-Encoder 精排
- 一般和 `search_by_hybrid()` 配合使用

参数：

- `query`：查询文本
- `candidates`：候选结果列表，通常来自 `search_by_hybrid()`
- `top_k`：返回前多少条，`None` 表示全部返回

示例：

```python
candidates = rag.search_by_hybrid("退款规则", top_k=10)
results = rag.rerank("退款规则", candidates, top_k=3)
```

### `search(query, top_k=3, vector_weight=0.7, bm25_weight=0.3, use_rerank=None, candidate_top_k=None)`

作用：

- 这是默认推荐的检索入口
- 内部会先做混合召回，再决定是否精排

参数：

- `query`：查询文本
- `top_k`：最终返回条数
- `vector_weight`：向量召回权重
- `bm25_weight`：BM25 权重
- `use_rerank`：是否启用精排；`None` 表示走初始化默认值
- `candidate_top_k`：先召回多少条候选再精排

推荐用法：

```python
results = rag.search(
    "退款规则",
    top_k=3,
    vector_weight=0.7,
    bm25_weight=0.3,
    use_rerank=True,
    candidate_top_k=10,
)
```

关闭精排：

```python
results = rag.search(
    "退款规则",
    top_k=3,
    use_rerank=False,
)
```

说明：

- 如果 `candidate_top_k` 小于 `top_k`，内部会自动至少取到 `top_k`
- `use_rerank=True` 时，返回的 `score` 就是 rerank 之后的分数

### `reset()`

作用：

- 清空当前知识库
- 重置 FAISS
- 清空 BM25
- 保留目录结构但覆盖索引和元数据文件

示例：

```python
rag.reset()
```

## 常见使用方式

### 1. 最简单的用法

```python
rag = RAGService()
rag.add_documents(["./docs/退款政策.txt"])
results = rag.search("退款规则", top_k=3)
```

### 2. 只看召回，不做精排

```python
rag = RAGService(default_use_rerank=False)
results = rag.search("售后政策", top_k=5)
```

### 3. 手动控制混合召回和精排

```python
rag = RAGService()

candidates = rag.search_by_hybrid(
    "如何申请退款",
    top_k=15,
    vector_weight=0.5,
    bm25_weight=0.5,
)

results = rag.rerank("如何申请退款", candidates, top_k=5)
```

### 4. 自定义 embedding 或 reranker

```python
rag = RAGService(
    embeddings=my_embeddings,
    reranker=my_reranker,
)
```

## 当前默认配置

- 向量模型：`text-embedding-v4`
- 精排模型：`BAAI/bge-reranker-v2-m3`
- 默认切片大小：`500`
- 默认切片重叠：`100`
- 默认混合检索权重：向量 `0.7`，BM25 `0.3`
- 默认精排：开启
- 默认候选数：`20`

## 注意事项

- 第一次调用 DashScope embedding 需要可用的 `DASHSCOPE_API_KEY`
- 第一次调用 rerank 需要下载 Cross-Encoder 模型
- PDF 抽取质量取决于原始 PDF 是否可提取文本
- 当前 metadata 只保存 `chunk_index`
- 当前实现适合中小规模知识库；数据量更大时可以考虑更复杂的索引或缓存策略
