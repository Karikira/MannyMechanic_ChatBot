"""
rag.py — RAG pipeline for Manny v4 (no ChromaDB)
=================================================
Uses sentence-transformers + numpy for in-memory vector search.
Fully compatible with Python 3.14 and Streamlit Cloud.
No ChromaDB dependency at all.
"""

import glob
import os
import numpy as np
from sentence_transformers import SentenceTransformer

from config import Config


class RAGPipeline:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = SentenceTransformer(cfg.embed_model)
        self._chunks: list[str] = []
        self._sources: list[str] = []
        self._embeddings = None  # numpy array, shape (N, dim)

    # ── Chunking ───────────────────────────────────────────────────────────
    def _chunk_text(self, text: str) -> list[str]:
        size    = self.cfg.chunk_size
        overlap = self.cfg.chunk_overlap
        chunks, start = [], 0
        while start < len(text):
            end   = start + size
            chunk = text[start:end]
            if end < len(text):
                break_at = max(chunk.rfind("\n"), chunk.rfind(". "))
                if break_at > size // 2:
                    chunk = text[start: start + break_at + 1]
                    end   = start + break_at + 1
            chunks.append(chunk.strip())
            start = end - overlap
        return [c for c in chunks if len(c) > 30]

    # ── Indexing ───────────────────────────────────────────────────────────
    def load_knowledge_base(self):
        if self._embeddings is not None:
            return  # already loaded this session

        txt_files = glob.glob(os.path.join(self.cfg.knowledge_dir, "*.txt"))
        for filepath in txt_files:
            filename = os.path.basename(filepath)
            text     = open(filepath, encoding="utf-8").read()
            for chunk in self._chunk_text(text):
                self._chunks.append(chunk)
                self._sources.append(filename)

        if self._chunks:
            self._embeddings = self._model.encode(
                self._chunks, convert_to_numpy=True, show_progress_bar=False
            )
            # Normalise for cosine similarity via dot product
            norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
            self._embeddings = self._embeddings / np.maximum(norms, 1e-10)

    # ── Retrieval ──────────────────────────────────────────────────────────
    def retrieve(self, query: str) -> tuple[str, list[str]]:
        if self._embeddings is None or len(self._chunks) == 0:
            return "", []

        q_vec = self._model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-10:
            q_vec = q_vec / q_norm

        scores  = (self._embeddings @ q_vec.T).flatten()
        k       = min(self.cfg.retriever_k, len(self._chunks))
        top_idx = np.argsort(scores)[::-1][:k]

        parts   = []
        sources = list({self._sources[i] for i in top_idx})
        for i in top_idx:
            parts.append(f"[Source: {self._sources[i]}]\n{self._chunks[i]}")

        return "\n\n---\n\n".join(parts), sources

    # ── Helpers ────────────────────────────────────────────────────────────
    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def inject_context(self, prompt: str, context: str) -> str:
        if not context:
            return prompt
        return prompt + f"\n\n=== RETRIEVED CONTEXT ===\n{context}\n=== END CONTEXT ===\n"
