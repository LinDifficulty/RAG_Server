# RAG Service

一个可用的 RAG 服务

## 功能

- 文档上传入库
- 文档切片
- 使用 FAISS 建立向量索引
- 使用 BM25 做关键词检索
- 使用混合检索融合向量分数和关键词分数
- 使用 Cross-Encoder 对候选结果精排
- 本地持久化索引和元数据
- 查询检索返回相关片段
- 嵌入模型使用 `DashScopeEmbeddings(model="text-embedding-v4")`

## 支持格式

- `.txt`
- `.md`
- `.pdf`

## 安装依赖

项目使用 `uv` 管理依赖，已补充：

- `faiss-cpu`
- `pypdf`
- `jieba`
- `langchain-community`
- `langchain-text-splitters`
- `rank-bm25`
- `dashscope`
- `sentence-transformers`

使用 `uv sync` 安装依赖
## 使用方式

```python
from rag_service import RAGService

rag = RAGService(data_dir="data")

rag.add_documents([
    "./docs/a.txt",
    "./docs/b.pdf",
])

rag.add_documents(
    ["./docs/c.md"],
    chunk_size=800,
    chunk_overlap=150,
)

results = rag.search("请帮我查找退款规则", top_k=3)
for item in results:
    print(item["score"], item["source"])
    print(item["content"])

vector_results = rag.search_by_vector("退款规则", top_k=3)
bm25_results = rag.search_by_bm25("退款规则", top_k=3)
hybrid_results = rag.search_by_hybrid("退款规则", top_k=10, vector_weight=0.7, bm25_weight=0.3)
rerank_results = rag.search(
    "退款规则",
    top_k=3,
    vector_weight=0.7,
    bm25_weight=0.3,
    use_rerank=True,
    candidate_top_k=10,
)
```

## 说明

- 向量模型默认使用 `text-embedding-v4`
- 使用前需要设置环境变量 `DASHSCOPE_API_KEY`
- 分片使用 `RecursiveCharacterTextSplitter`，`chunk_size` 和 `chunk_overlap` 都可以传参覆盖
- `search()` 默认先做混合召回，再做 Cross-Encoder 精排
- 混合召回可以直接调用 `search_by_hybrid()`
- 关键词检索底层使用 `BM25Plus`，这样在小语料场景下分数会更稳定
- 精排模型默认使用 `BAAI/bge-reranker-v2-m3`
- 第一次触发精排时会下载 Cross-Encoder 模型，因此首轮可能会慢一些
- `candidate_top_k` 表示先召回多少候选片段再进入精排
- 返回结果里：
- `score` 是当前阶段最终分数
- `hybrid_score` 是混合召回分数
- `rerank_score` 是 Cross-Encoder 精排分数；如果没开精排则为 `None`
- 索引文件默认保存在 `./data/faiss.index`
- 元数据默认保存在 `./data/metadata.json`
- `reset()` 可以清空当前索引
