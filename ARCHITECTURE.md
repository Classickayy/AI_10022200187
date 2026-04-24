# Part F: Architecture & System Design

## 1) Architecture Diagram

```text
                         OFFLINE / INDEXING FLOW
  Ghana_Election_Result.csv + 2025 Budget PDF
                    |
                    v
         DataProcessor (cleaning + chunking)
                    |
                    v
      Embedding Pipeline (SentenceTransformer)
                    |
                    v
             FAISS Vector Index + Chunk Metadata


                          ONLINE / INFERENCE FLOW
User Query (Streamlit UI)
          |
          v
AdvancedRetriever
  - top-k vector search
  - hybrid keyword + vector scoring
  - optional query expansion / reranking
          |
          v
PromptManager
  - context ranking/filtering/truncation
  - hallucination-control template
          |
          v
LLM (OpenAI or GROK endpoint)
          |
          v
Final Response + Sources + Similarity Scores + Prompt Preview
          |
          v
Stage Logs (retrieval, context_selection, prompt, generation)
```

## 2) Data Flow and Component Interaction

- `src/data_engineering.py` loads CSV/PDF, cleans noisy text, and chunks documents.
- `src/embeddings.py` converts chunks into embeddings and stores them in FAISS.
- `src/retrieval.py` retrieves top-k chunks using similarity scoring and hybrid search.
- `src/prompt_manager.py` builds prompts by injecting selected context and hallucination constraints.
- `src/rag_pipeline.py` orchestrates the full flow and logs each stage.
- `app.py` is the UI layer that shows retrieved documents, similarity scores, final prompt, and response.

## 3) Why This Design Fits the Domain

- The domain (election results + budget statements) is fact-heavy and source-based, so RAG grounding is more suitable than pure LLM recall.
- FAISS retrieval is efficient for repeated question answering over a fixed document set.
- Hybrid retrieval improves robustness for both numeric/election-style lookups and narrative budget sections.
- Prompt-level hallucination control is important because users may ask ambiguous or misleading policy questions.
- Stage logging supports auditability, which is useful for academic evaluation and debugging model behavior.