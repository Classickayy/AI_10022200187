import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from a local .env file if present (do NOT commit .env)
load_dotenv()

@dataclass
class Config:
    # Data paths
    CSV_PATH = "data/raw/Ghana_Election_Result.csv"
    PDF_PATH = "data/raw/2025-Budget-Statement-and-Economic-Policy_v4.pdf"
    PROCESSED_DIR = "data/processed"
    VECTOR_DIR = "vector_store"
    LOG_DIR = "logs"
    
    # Chunking parameters (Part A justification included in docs)
    CHUNK_SIZE = 512          # Tokens (optimal for semantic coherence)
    CHUNK_OVERLAP = 50        # 10% overlap to preserve context boundaries
    MAX_CONTEXT_LENGTH = 2048 # Token budget for LLM context
    
    # Model settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    LLM_MODEL = "llama-3.3-70b-versatile"  # or "gpt-4" if budget allows
    
    # Retrieval settings
    TOP_K = 5
    RERANK_TOP_K = 3
    
    # API Keys (use environment variables in production)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # LLM provider selection: 'openai' or 'groq' (default 'openai' if OPENAI_API_KEY present)
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    # Groq provider configuration — set these via environment variables
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    # Fallback to legacy GROK for backward compatibility
    GROK_API_KEY = os.getenv("GROK_API_KEY")