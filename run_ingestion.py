"""
run_ingestion.py
----------------
One-time script to:
  1. Load & clean both raw datasets (CSV + PDF)
  2. Chunk them using the recursive sentence-aware strategy
  3. Embed the chunks and persist a FAISS vector index

Run this once (or whenever the raw data changes) before starting app.py.
Usage:
    python run_ingestion.py
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_engineering import DataProcessor
from src.embeddings import CustomEmbeddingPipeline


def main():
    print("=" * 60)
    print("  RAG Ingestion Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Step 1: Load, clean, and chunk both datasets
    # ------------------------------------------------------------------ #
    print("\n[1/3] Loading and chunking datasets...")
    processor = DataProcessor()

    try:
        chunks = processor.process_all(strategy="recursive")
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Make sure both raw files exist in data/raw/:")
        print("  - Ghana_Election_Result.csv")
        print("  - 2025-Budget-Statement-and-Economic-Policy_v4.pdf")
        sys.exit(1)

    print(f"   ✅ Total chunks produced : {len(chunks)}")

    # Count by source for a quick sanity check
    csv_count = sum(1 for c in chunks if c.get("source") == "election_data")
    pdf_count = sum(1 for c in chunks if c.get("source") == "budget_2025")
    print(f"      • Election CSV chunks : {csv_count}")
    print(f"      • Budget PDF chunks   : {pdf_count}")

    # Show a sample chunk from each source
    csv_sample = next((c for c in chunks if c.get("source") == "election_data"), None)
    pdf_sample = next((c for c in chunks if c.get("source") == "budget_2025"), None)

    if csv_sample:
        print(f"\n   CSV sample chunk:\n   {csv_sample['content'][:200]!r}")
    if pdf_sample:
        print(f"\n   PDF sample chunk:\n   {pdf_sample['content'][:200]!r}")

    # ------------------------------------------------------------------ #
    # Step 2: Embed chunks and build FAISS index
    # ------------------------------------------------------------------ #
    print("\n[2/3] Embedding chunks and building FAISS index...")
    print("      (This may take a few minutes for the PDF — please wait)")

    embedder = CustomEmbeddingPipeline()
    embedder.build_faiss_index(chunks)

    print(f"   ✅ FAISS index built with {embedder.index.ntotal} vectors")
    print(f"   ✅ Saved to : vector_store/faiss_index.bin")
    print(f"   ✅ Metadata : vector_store/chunks_metadata.pkl")

    # ------------------------------------------------------------------ #
    # Step 3: Quick retrieval smoke-test
    # ------------------------------------------------------------------ #
    print("\n[3/3] Running smoke-test queries...")

    test_queries = [
        "Who won the most votes in Greater Accra?",
        "What is the 2025 education budget allocation?",
    ]

    for q in test_queries:
        results = embedder.similarity_search(q, k=3)
        print(f"\n   Query : {q!r}")
        for rank, (doc, score) in enumerate(results, 1):
            preview = doc["content"][:100].replace("\n", " ")
            print(f"   [{rank}] score={score:.4f} | source={doc['source']} | {preview!r}")

    print("\n" + "=" * 60)
    print("  ✅ Ingestion complete — app.py / Groq is ready to use the index!")
    print("=" * 60)


if __name__ == "__main__":
    main()
