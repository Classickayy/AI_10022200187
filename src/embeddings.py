import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from typing import List, Tuple, Dict
from config import Config

class CustomEmbeddingPipeline:
    def __init__(self):
        self.config = Config()
        self.model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
        
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Manual embedding without LangChain"""
        return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    def build_faiss_index(self, chunks: List[Dict]):
        """Custom vector storage using FAISS (no pre-built pipelines)"""
        self.chunks = chunks
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embed_documents(texts)
        
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Create FAISS index (Inner Product = Cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Save index
        os.makedirs(self.config.VECTOR_DIR, exist_ok=True)
        faiss.write_index(self.index, f"{self.config.VECTOR_DIR}/faiss_index.bin")
        
        with open(f"{self.config.VECTOR_DIR}/chunks_metadata.pkl", 'wb') as f:
            pickle.dump(chunks, f)
            
    def load_index(self):
        """Load existing index if present. If missing, attempt to rebuild
        from saved chunk metadata. If no metadata exists, leave index as None.
        """
        index_path = os.path.join(self.config.VECTOR_DIR, "faiss_index.bin")
        chunks_path = os.path.join(self.config.VECTOR_DIR, "chunks_metadata.pkl")

        # Ensure vector directory exists
        os.makedirs(self.config.VECTOR_DIR, exist_ok=True)

        if os.path.exists(index_path):
            try:
                self.index = faiss.read_index(index_path)
            except Exception as e:
                print(f"Warning: failed to load FAISS index: {e}")
                self.index = None

        # Load chunks metadata if available
        if os.path.exists(chunks_path):
            try:
                with open(chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
            except Exception as e:
                print(f"Warning: failed to load chunks metadata: {e}")
                self.chunks = []

        # If index missing but we have chunks, rebuild index
        if self.index is None and self.chunks:
            try:
                print("Rebuilding FAISS index from chunks metadata...")
                self.build_faiss_index(self.chunks)
            except Exception as e:
                print(f"Error rebuilding FAISS index: {e}")
                self.index = None
        # If neither exists, leave index as None and proceed (retrieval will return empty)
    
    def similarity_search(self, query: str, k: int = None) -> List[Tuple[Dict, float]]:
        """Top-k retrieval with similarity scoring"""
        k = k or self.config.TOP_K
        query_vector = self.model.encode([query])
        query_vector = query_vector / np.linalg.norm(query_vector)
        # If no index is available, return empty results
        if self.index is None:
            return []

        scores, indices = self.index.search(query_vector.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results