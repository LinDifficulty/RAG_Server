from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import jieba
import numpy as np
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from rank_bm25 import BM25Plus
from sentence_transformers import CrossEncoder

# 支持的文档格式集合，仅处理这些扩展名的文件。
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


class RAGService:
    """轻量 RAG 服务。

    支持：
    1. DashScope 向量化 + FAISS 向量检索
    2. BM25 关键词检索
    3. 向量 + BM25 混合召回
    4. Cross-Encoder 精排
    """

    def __init__(
        self,
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
    ) -> None:
        # 所有索引和元数据都保存在 data_dir，方便持久化复用。
        self.base_dir = Path(data_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.base_dir / "faiss.index"
        self.metadata_path = self.base_dir / "metadata.json"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 向量模型默认使用 DashScope，也支持外部传入自定义 embeddings。
        self.embeddings = embeddings or DashScopeEmbeddings(model=model_name)

        # 重排序模型默认使用多语言 reranker，首次调用时再懒加载。
        self.reranker_model_name = reranker_model_name
        self.reranker = reranker
        self.reranker_device = reranker_device
        self.reranker_batch_size = reranker_batch_size
        self.default_use_rerank = default_use_rerank
        self.default_candidate_top_k = default_candidate_top_k

        self.records = self._load_records()
        self.bm25: BM25Plus | None = None
        self.index = self._load_or_create_index()
        self._rebuild_bm25()

    def add_documents(
        self,
        file_paths: list[str],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> dict:
        """读取文档、切片并写入 FAISS，同时刷新 BM25 索引。"""
        actual_chunk_size = chunk_size or self.chunk_size
        actual_chunk_overlap = (
            self.chunk_overlap if chunk_overlap is None else chunk_overlap
        )
        if actual_chunk_overlap >= actual_chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        records: list[dict] = []
        for file_path in file_paths:
            path = Path(file_path)
            text = self._read_document(path)
            for chunk_index, content in enumerate(
                self._split_text(text, actual_chunk_size, actual_chunk_overlap)
            ):
                records.append(
                    {
                        "source": str(path),
                        "content": content,
                        "metadata": {"chunk_index": chunk_index},
                    }
                )

        if not records:
            return {"added_chunks": 0, "sources": []}

        vectors = self._embed_texts([record["content"] for record in records])
        self.index.add(vectors)
        self.records.extend(records)
        self._rebuild_bm25()
        self._persist()

        return {
            "added_chunks": len(records),
            "sources": sorted({record["source"] for record in records}),
        }

    def search_by_vector(self, query: str, top_k: int = 3) -> list[dict]:
        """仅使用向量召回。"""
        if not query.strip() or not self.records or self.index.ntotal == 0:
            return []

        limit = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(self._embed_texts([query]), limit)

        results = []
        for raw_score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            vector_score = self._normalize_vector_score(float(raw_score))
            results.append(
                self._build_result(
                    idx=int(idx),
                    score=vector_score,
                    vector_score=vector_score,
                    bm25_score=0.0,
                    hybrid_score=vector_score,
                    rerank_score=None,
                    retrieval_mode="vector",
                )
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results

    def search_by_bm25(self, query: str, top_k: int = 3) -> list[dict]:
        """仅使用 BM25 关键词检索。"""
        if not query.strip() or not self.records or self.bm25 is None:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        raw_scores = np.asarray(self.bm25.get_scores(query_tokens), dtype="float32")
        if raw_scores.size == 0 or float(raw_scores.max()) <= 0:
            return []

        normalized_scores = self._normalize_bm25_scores(raw_scores)
        top_indices = np.argsort(raw_scores)[::-1][: min(top_k, len(raw_scores))]

        results = []
        for idx in top_indices:
            if raw_scores[idx] <= 0:
                continue
            bm25_score = float(normalized_scores[idx])
            results.append(
                self._build_result(
                    idx=int(idx),
                    score=bm25_score,
                    vector_score=0.0,
                    bm25_score=bm25_score,
                    hybrid_score=bm25_score,
                    rerank_score=None,
                    retrieval_mode="bm25",
                )
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results

    def search_by_hybrid(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> list[dict]:
        """向量召回和 BM25 召回融合后的结果，不做精排。"""
        if not query.strip() or not self.records:
            return []

        self._validate_weights(vector_weight, bm25_weight)

        limit = min(top_k, len(self.records))
        vector_scores = self._vector_score_map(query, limit)
        bm25_scores = self._bm25_score_map(query, limit)
        candidate_ids = set(vector_scores) | set(bm25_scores)

        ranked_results = []
        for idx in candidate_ids:
            vector_score = vector_scores.get(idx, 0.0)
            bm25_score = bm25_scores.get(idx, 0.0)
            hybrid_score = vector_weight * vector_score + bm25_weight * bm25_score
            ranked_results.append(
                self._build_result(
                    idx=idx,
                    score=hybrid_score,
                    vector_score=vector_score,
                    bm25_score=bm25_score,
                    hybrid_score=hybrid_score,
                    rerank_score=None,
                    retrieval_mode="hybrid",
                )
            )

        ranked_results.sort(key=lambda item: item["score"], reverse=True)
        return ranked_results[:top_k]

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """使用 Cross-Encoder 对候选结果做精排。"""
        if not query.strip() or not candidates:
            return []

        reranker = self._get_reranker()
        pairs = [(query, item["content"]) for item in candidates]
        raw_scores = reranker.predict(
            pairs,
            batch_size=self.reranker_batch_size,
            show_progress_bar=False,
        )
        rerank_scores = self._coerce_rerank_scores(raw_scores)

        reranked = []
        for item, rerank_score in zip(candidates, rerank_scores, strict=False):
            merged = dict(item)
            merged["score"] = float(rerank_score)
            merged["rerank_score"] = float(rerank_score)
            merged["retrieval_mode"] = "hybrid_rerank"
            reranked.append(merged)

        reranked.sort(key=lambda item: item["score"], reverse=True)
        if top_k is None:
            return reranked
        return reranked[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 3,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        use_rerank: bool | None = None,
        candidate_top_k: int | None = None,
    ) -> list[dict]:
        """默认搜索入口。

        先做混合召回，再按配置决定是否执行 Cross-Encoder 精排。
        """
        if not query.strip() or not self.records:
            return []

        actual_use_rerank = (
            self.default_use_rerank if use_rerank is None else use_rerank
        )
        actual_candidate_top_k = candidate_top_k or self.default_candidate_top_k
        actual_candidate_top_k = max(actual_candidate_top_k, top_k)

        hybrid_candidates = self.search_by_hybrid(
            query=query,
            top_k=actual_candidate_top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )

        if not actual_use_rerank:
            return hybrid_candidates[:top_k]

        return self.rerank(query=query, candidates=hybrid_candidates, top_k=top_k)

    def reset(self) -> None:
        """清空 FAISS、BM25 和元数据。"""
        self.records = []
        self.index = self._create_index()
        self._rebuild_bm25()
        self._persist()

    def _read_document(self, path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        if path.suffix.lower() in {".txt", ".md"}:
            return path.read_text(encoding="utf-8").strip()

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

    def _split_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:
        if not text:
            return []

        # 优先按段落和中文标点切，能比纯字符切片保留更多自然语义边界。
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        return splitter.split_text(text)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = self.embeddings.embed_documents(texts)
        matrix = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        return matrix

    def _load_or_create_index(self) -> faiss.Index:
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))
        return self._create_index()

    def _create_index(self) -> faiss.Index:
        dimension = len(self.embeddings.embed_query("test"))
        return faiss.IndexFlatIP(dimension)

    def _rebuild_bm25(self) -> None:
        # BM25 不单独持久化，启动时根据已有 chunk 重建即可。
        tokenized_corpus = [
            self._tokenize(record["content"])
            for record in self.records
            if record["content"].strip()
        ]
        # 小语料场景下，BM25Plus 比 BM25Okapi 更容易得到稳定的关键词分数。
        self.bm25 = BM25Plus(tokenized_corpus) if tokenized_corpus else None

    def _tokenize(self, text: str) -> list[str]:
        # jieba 对中文关键词检索比简单 split 更合适。
        return [
            token.lower().strip()
            for token in jieba.lcut_for_search(text)
            if token.strip()
        ]

    def _vector_score_map(self, query: str, limit: int) -> dict[int, float]:
        if self.index.ntotal == 0:
            return {}

        scores, indices = self.index.search(self._embed_texts([query]), limit)
        result: dict[int, float] = {}
        for raw_score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            result[int(idx)] = self._normalize_vector_score(float(raw_score))
        return result

    def _bm25_score_map(self, query: str, limit: int) -> dict[int, float]:
        if self.bm25 is None:
            return {}

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return {}

        raw_scores = np.asarray(self.bm25.get_scores(query_tokens), dtype="float32")
        if raw_scores.size == 0 or float(raw_scores.max()) <= 0:
            return {}

        normalized_scores = self._normalize_bm25_scores(raw_scores)
        top_indices = np.argsort(raw_scores)[::-1][:limit]
        return {
            int(idx): float(normalized_scores[idx])
            for idx in top_indices
            if raw_scores[idx] > 0
        }

    def _normalize_vector_score(self, score: float) -> float:
        # 内积分数近似落在 [-1, 1]，缩放到 [0, 1] 后更方便与 BM25 融合。
        return max(0.0, min(1.0, (score + 1) / 2))

    def _normalize_bm25_scores(self, scores: np.ndarray) -> np.ndarray:
        max_score = float(scores.max())
        if max_score <= 0:
            return np.zeros_like(scores)
        return scores / max_score

    def _validate_weights(self, vector_weight: float, bm25_weight: float) -> None:
        if vector_weight < 0 or bm25_weight < 0:
            raise ValueError("vector_weight and bm25_weight must be non-negative")
        if vector_weight == 0 and bm25_weight == 0:
            raise ValueError("At least one retrieval weight must be greater than 0")

    def _get_reranker(self) -> Any:
        # 懒加载 reranker，避免只做入库时也下载和初始化 Cross-Encoder。
        if self.reranker is None:
            self.reranker = CrossEncoder(
                self.reranker_model_name,
                device=self.reranker_device,
                trust_remote_code=True,
            )
        return self.reranker

    def _coerce_rerank_scores(self, scores: Any) -> list[float]:
        array = np.asarray(scores, dtype="float32")
        if array.ndim == 0:
            return [float(array)]
        if array.ndim == 1:
            return [float(item) for item in array.tolist()]

        # 某些模型可能返回多维输出，这里默认取最后一列作为相关性分数。
        return [float(item) for item in array[:, -1].tolist()]

    def _build_result(
        self,
        idx: int,
        score: float,
        vector_score: float,
        bm25_score: float,
        hybrid_score: float,
        rerank_score: float | None,
        retrieval_mode: str,
    ) -> dict:
        record = self.records[idx]
        return {
            "score": float(score),
            "vector_score": float(vector_score),
            "bm25_score": float(bm25_score),
            "hybrid_score": float(hybrid_score),
            "rerank_score": None if rerank_score is None else float(rerank_score),
            "content": record["content"],
            "source": record["source"],
            "metadata": record["metadata"],
            "retrieval_mode": retrieval_mode,
        }

    def _load_records(self) -> list[dict]:
        if not self.metadata_path.exists():
            return []
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def _persist(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
