import numpy as np
from fastembed import TextEmbedding   # ONNX-based, no PyTorch required (~50 MB)
import faiss
import pickle
import os
from typing import List, Tuple, Dict
from config import Config

class CustomEmbeddingPipeline:
    def __init__(self):
        self.config = Config()
        # Uses ONNX Runtime — much lighter than sentence-transformers + torch
        # "BAAI/bge-small-en-v1.5" is supported in fastembed 0.5+
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        # Embedding dimension for bge-small-en-v1.5
        self.dimension = 384
        self.index = None
        self.chunks = []

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts using fastembed (ONNX, no PyTorch)."""
        embeddings = list(self.model.embed(texts))
        return np.array(embeddings, dtype="float32")

    def build_faiss_index(self, chunks: List[Dict]):
        """Custom vector storage using FAISS (no pre-built pipelines)."""
        self.chunks = chunks
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embed_documents(texts)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)   # avoid div-by-zero
        embeddings = embeddings / norms

        # Create FAISS index (Inner Product = Cosine similarity for normalised vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        # Persist index
        os.makedirs(self.config.VECTOR_DIR, exist_ok=True)
        faiss.write_index(self.index, f"{self.config.VECTOR_DIR}/faiss_index.bin")

        with open(f"{self.config.VECTOR_DIR}/chunks_metadata.pkl", 'wb') as f:
            pickle.dump(chunks, f)

    def load_index(self):
        """Load existing index if present; rebuild from chunk metadata if index missing."""
        index_path = os.path.join(self.config.VECTOR_DIR, "faiss_index.bin")
        chunks_path = os.path.join(self.config.VECTOR_DIR, "chunks_metadata.pkl")

        os.makedirs(self.config.VECTOR_DIR, exist_ok=True)

        if os.path.exists(index_path):
            try:
                self.index = faiss.read_index(index_path)
            except Exception as e:
                print(f"Warning: failed to load FAISS index: {e}")
                self.index = None

        if os.path.exists(chunks_path):
            try:
                with open(chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
            except Exception as e:
                print(f"Warning: failed to load chunks metadata: {e}")
                self.chunks = []

        # If index is missing but chunks exist, rebuild
        if self.index is None and self.chunks:
            try:
                print("Rebuilding FAISS index from chunks metadata...")
                self.build_faiss_index(self.chunks)
            except Exception as e:
                print(f"Error rebuilding FAISS index: {e}")
                self.index = None

    def similarity_search(self, query: str, k: int = None) -> List[Tuple[Dict, float]]:
        """Top-k retrieval with cosine similarity scoring."""
        k = k or self.config.TOP_K

        if self.index is None:
            return []

        query_vec = np.array(list(self.model.embed([query])), dtype="float32")
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results