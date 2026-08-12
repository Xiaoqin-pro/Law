"""Phase 2 retrieval baselines for Chinese legal statute retrieval.

The module deliberately keeps the retrieval layers independent:

* :class:`BM25Index` is a small, cacheable inverted-index implementation;
* :class:`DenseEmbedder` uses the Hugging Face Transformers API directly so the
  project does not depend on a second embedding framework;
* :class:`DenseIndex` wraps an exact FAISS inner-product index;
* :func:`reciprocal_rank_fusion` combines two ranked lists without looking at
  gold labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import jieba
import numpy as np


TOKEN_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z]+|\d+(?:\.\d+)?")


def chinese_tokens(text: str) -> List[str]:
    """Tokenize Chinese legal text into words plus useful atomic symbols.

    Jieba supplies word-level tokens for legal phrases; single Chinese
    characters are retained as a fallback for names/article numbers that are
    split differently by the tokenizer. Punctuation and whitespace are
    removed. The deterministic tokenization is part of the experiment config.
    """

    tokens: List[str] = []
    for chunk in jieba.lcut(str(text), cut_all=False):
        chunk = chunk.strip()
        if not chunk:
            continue
        pieces = TOKEN_PATTERN.findall(chunk)
        if not pieces:
            continue
        if len(pieces) == 1 and pieces[0] == chunk:
            tokens.append(chunk.lower())
        else:
            tokens.extend(piece.lower() for piece in pieces)
    return tokens


def corpus_fingerprint(corpus: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in corpus:
        digest.update(str(row["statute_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["statute_name"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["statute_text"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def query_fingerprint(queries: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in queries:
        digest.update(str(row["query_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["question"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class BM25Index:
    """A cacheable BM25 inverted index over statute name + text."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_ids: List[int] = []
        self.doc_names: List[str] = []
        self.doc_lengths: np.ndarray = np.array([], dtype=np.float32)
        self.avgdl = 0.0
        self.postings: Dict[str, Dict[int, int]] = {}
        self.idf: Dict[str, float] = {}
        self.fingerprint = ""

    @staticmethod
    def _document_text(row: Mapping[str, Any]) -> str:
        return f"{row['statute_name']} {row['statute_text']}"

    def build(self, corpus: Sequence[Mapping[str, Any]]) -> "BM25Index":
        self.doc_ids = [int(row["statute_id"]) for row in corpus]
        self.doc_names = [str(row["statute_name"]) for row in corpus]
        self.fingerprint = corpus_fingerprint(corpus)
        postings: Dict[str, Dict[int, int]] = defaultdict(dict)
        lengths: List[int] = []
        for doc_index, row in enumerate(corpus):
            counts = Counter(chinese_tokens(self._document_text(row)))
            lengths.append(sum(counts.values()))
            for token, frequency in counts.items():
                postings[token][doc_index] = frequency
        self.postings = dict(postings)
        self.doc_lengths = np.asarray(lengths, dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_lengths)) if lengths else 0.0
        document_count = len(self.doc_ids)
        self.idf = {
            token: math.log(1.0 + (document_count - len(doc_postings) + 0.5) / (len(doc_postings) + 0.5))
            for token, doc_postings in self.postings.items()
        }
        return self

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids,
            "doc_names": self.doc_names,
            "doc_lengths": self.doc_lengths,
            "avgdl": self.avgdl,
            "postings": self.postings,
            "idf": self.idf,
            "fingerprint": self.fingerprint,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        index = cls(k1=payload["k1"], b=payload["b"])
        index.doc_ids = payload["doc_ids"]
        index.doc_names = payload["doc_names"]
        index.doc_lengths = payload["doc_lengths"]
        index.avgdl = payload["avgdl"]
        index.postings = payload["postings"]
        index.idf = payload["idf"]
        index.fingerprint = payload["fingerprint"]
        return index

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        scores: Dict[int, float] = defaultdict(float)
        query_counts = Counter(chinese_tokens(query))
        if self.avgdl <= 0:
            return []
        for token, query_frequency in query_counts.items():
            del query_frequency  # BM25 query term frequency is intentionally binary here.
            doc_postings = self.postings.get(token)
            if not doc_postings:
                continue
            idf = self.idf[token]
            for doc_index, term_frequency in doc_postings.items():
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * float(self.doc_lengths[doc_index]) / self.avgdl
                )
                scores[doc_index] += idf * term_frequency * (self.k1 + 1.0) / denominator
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [
            {
                "rank": rank,
                "score": float(score),
                "statute_id": self.doc_ids[doc_index],
                "statute_name": self.doc_names[doc_index],
            }
            for rank, (doc_index, score) in enumerate(ranked, start=1)
        ]


class DenseEmbedder:
    """BGE-M3 encoder with GPU/CPU selection and batch-size backoff."""

    def __init__(
        self,
        model_name: str,
        *,
        max_length: int = 512,
        batch_size: int = 8,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.normalize_embeddings = bool(normalize_embeddings)
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir) if cache_dir else None)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=str(cache_dir) if cache_dir else None)
        self.model.to(self.device)
        self.model.eval()
        self.model_revision = getattr(getattr(self.model, "config", None), "_commit_hash", None)
        if self.model_revision is None:
            self.model_revision = "unknown"

    def _pool(self, model_output: Any, attention_mask: Any) -> Any:
        embeddings = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        if self.normalize_embeddings:
            pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled

    def encode(self, texts: Sequence[str], *, batch_size: Optional[int] = None) -> np.ndarray:
        if not texts:
            hidden = int(getattr(self.model.config, "hidden_size", 0))
            return np.empty((0, hidden), dtype=np.float32)
        current_batch_size = max(1, int(batch_size or self.batch_size))
        chunks: List[np.ndarray] = []
        position = 0
        while position < len(texts):
            current = min(current_batch_size, len(texts) - position)
            batch_texts = list(texts[position : position + current])
            try:
                with self.torch.inference_mode():
                    encoded = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    encoded = {key: value.to(self.device) for key, value in encoded.items()}
                    output = self.model(**encoded)
                    pooled = self._pool(output, encoded["attention_mask"])
                    chunks.append(pooled.detach().float().cpu().numpy())
                position += current
            except RuntimeError as exc:
                is_oom = "out of memory" in str(exc).lower() or "cuda" in str(exc).lower() and "memory" in str(exc).lower()
                if not is_oom or current_batch_size == 1:
                    raise
                if self.device.type == "cuda":
                    self.torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)
        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

    def encode_corpus_cached(
        self,
        corpus: Sequence[Mapping[str, Any]],
        cache_dir: Path,
        *,
        batch_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = corpus_fingerprint(corpus)
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.model_name)
        prefix = cache_dir / f"{safe_model}_{fingerprint[:12]}"
        vector_path = prefix.with_suffix(".npy")
        manifest_path = prefix.with_name(prefix.name + ".manifest.json")
        progress_path = prefix.with_name(prefix.name + ".progress.json")
        texts = [f"{row['statute_name']} {row['statute_text']}" for row in corpus]
        manifest: Dict[str, Any] = {
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "corpus_fingerprint": fingerprint,
            "count": len(texts),
            "max_length": self.max_length,
            "normalize_embeddings": self.normalize_embeddings,
        }
        if vector_path.exists() and manifest_path.exists():
            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if all(saved_manifest.get(key) == value for key, value in manifest.items()):
                return np.load(vector_path, mmap_mode="r"), {**manifest, "cache_hit": True, "vector_path": str(vector_path)}

        completed = 0
        dimension: Optional[int] = None
        if progress_path.exists() and vector_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("manifest") == manifest:
                completed = int(progress.get("completed", 0))
                dimension = int(progress["dimension"])

        if dimension is None:
            first = self.encode(texts[:1], batch_size=1)
            dimension = int(first.shape[1])
            matrix = np.lib.format.open_memmap(vector_path, mode="w+", dtype="float32", shape=(len(texts), dimension))
            if len(texts):
                matrix[0] = first[0]
            completed = 1
            matrix.flush()
            progress_path.write_text(json.dumps({"manifest": manifest, "completed": completed, "dimension": dimension}, indent=2), encoding="utf-8")
        else:
            matrix = np.lib.format.open_memmap(vector_path, mode="r+", dtype="float32", shape=(len(texts), dimension))

        while completed < len(texts):
            end = min(completed + int(batch_size or self.batch_size), len(texts))
            matrix[completed:end] = self.encode(texts[completed:end], batch_size=batch_size)
            completed = end
            matrix.flush()
            progress_path.write_text(json.dumps({"manifest": manifest, "completed": completed, "dimension": dimension}, indent=2), encoding="utf-8")
        matrix.flush()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        progress_path.unlink(missing_ok=True)
        return np.load(vector_path, mmap_mode="r"), {**manifest, "cache_hit": False, "vector_path": str(vector_path)}


class DenseIndex:
    """Exact FAISS index over normalized dense vectors."""

    def __init__(self, vectors: np.ndarray, corpus: Sequence[Mapping[str, Any]]) -> None:
        import faiss

        self.faiss = faiss
        self.corpus = corpus
        self.index = faiss.IndexFlatIP(int(vectors.shape[1]))
        self.index.add(np.asarray(vectors, dtype=np.float32))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        self.faiss.write_index(self.index, str(temporary))
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path, corpus: Sequence[Mapping[str, Any]]) -> "DenseIndex":
        instance = cls.__new__(cls)
        import faiss

        instance.faiss = faiss
        instance.corpus = corpus
        instance.index = faiss.read_index(str(path))
        return instance

    def search(self, vector: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        scores, indices = self.index.search(np.asarray(vector, dtype=np.float32).reshape(1, -1), top_k)
        results: List[Dict[str, Any]] = []
        for rank, (score, doc_index) in enumerate(zip(scores[0], indices[0]), start=1):
            if int(doc_index) < 0:
                continue
            row = self.corpus[int(doc_index)]
            results.append({
                "rank": rank,
                "score": float(score),
                "statute_id": int(row["statute_id"]),
                "statute_name": str(row["statute_name"]),
            })
        return results


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Mapping[str, Any]]],
    *,
    rrf_k: int = 60,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    scores: Dict[int, float] = defaultdict(float)
    metadata: Dict[int, Mapping[str, Any]] = {}
    for ranked in ranked_lists:
        for result in ranked:
            statute_id = int(result["statute_id"])
            scores[statute_id] += 1.0 / (rrf_k + int(result["rank"]))
            metadata.setdefault(statute_id, result)
    ordered = sorted(scores, key=lambda sid: (-scores[sid], sid))[:top_k]
    return [
        {
            "rank": rank,
            "score": float(scores[statute_id]),
            "statute_id": statute_id,
            "statute_name": str(metadata[statute_id]["statute_name"]),
        }
        for rank, statute_id in enumerate(ordered, start=1)
    ]


def retrieval_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    cutoffs: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    if not rows:
        return {f"recall_at_{cutoff}": 0.0 for cutoff in cutoffs} | {"mrr": 0.0}
    totals = {f"recall_at_{cutoff}": 0.0 for cutoff in cutoffs}
    reciprocal_ranks: List[float] = []
    for row in rows:
        gold = {int(value) for value in row["gold_statute_ids"]}
        ranked = [int(result["statute_id"]) for result in row["results"]]
        for cutoff in cutoffs:
            totals[f"recall_at_{cutoff}"] += float(bool(gold.intersection(ranked[:cutoff])))
        rank = next((index + 1 for index, statute_id in enumerate(ranked) if statute_id in gold), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    count = float(len(rows))
    return {key: round(value / count, 6) for key, value in totals.items()} | {"mrr": round(sum(reciprocal_ranks) / count, 6)}


def error_analysis(
    result_by_retriever: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    by_query: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    for retriever, rows in result_by_retriever.items():
        for row in rows:
            by_query.setdefault(retriever, {})[int(row["query_id"])] = row
    retrievers = list(result_by_retriever)
    if not {"bm25", "dense", "hybrid"}.issubset(by_query):
        return {}
    categories: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for query_id in sorted(set(by_query["bm25"]) | set(by_query["dense"]) | set(by_query["hybrid"])):
        rows = {name: by_query[name][query_id] for name in retrievers if query_id in by_query[name]}
        success = {
            name: bool(set(int(v) for v in row["gold_statute_ids"]).intersection(int(r["statute_id"]) for r in row["results"]))
            for name, row in rows.items()
        }
        payload = {
            "query_id": query_id,
            "question": rows["bm25"]["question"],
            "gold_statute_ids": rows["bm25"]["gold_statute_ids"],
            "success_at_10": success,
        }
        if success["bm25"] and not success["dense"]:
            categories["bm25_success_dense_failure"].append(payload)
        if success["dense"] and not success["bm25"]:
            categories["dense_success_bm25_failure"].append(payload)
        if not success["bm25"] and not success["dense"]:
            categories["both_bm25_dense_failure"].append(payload)
        if success["hybrid"] and not success["bm25"] and not success["dense"]:
            categories["hybrid_rescue"].append(payload)
        if success["hybrid"] is False and (success["bm25"] or success["dense"]):
            categories["hybrid_regression"].append(payload)
    return {key: values for key, values in categories.items()}


def now_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
