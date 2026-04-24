# 2-Minute Video Walkthrough Guide

Use this as a speaking guide while screen-recording. Keep total time under 2 minutes.

## 0:00 - 0:20 (Problem + Goal)
- "This project builds a RAG assistant over Ghana election and 2025 budget data."
- "Goal: answer with grounded evidence, reduce hallucinations, and show full pipeline transparency."

## 0:20 - 0:45 (Architecture)
- Open `ARCHITECTURE.md`.
- Say: "Offline flow: clean/chunk data -> embeddings -> FAISS index."
- Say: "Online flow: query -> retrieval -> context selection -> prompt -> LLM -> response + logs."

## 0:45 - 1:10 (Live App Features)
- Open app (`streamlit run app.py`).
- Show:
  - query input box
  - retrieved chunks with source + similarity score
  - final prompt expander
  - final response panel

## 1:10 - 1:35 (Design Decisions)
- "I used hybrid retrieval (vector + keyword), optional expansion/reranking, and context-window control in prompt manager."
- "Prompt includes hallucination control with uncertainty and evidence formatting."
- "I added stage logging for retrieval, context selection, prompt construction, and generation."

## 1:35 - 1:55 (Innovation + Evaluation)
- "Innovation: feedback memory boosts retrieval ranking for similar future queries with positive feedback."
- Open `MANUAL_EXPERIMENT_LOG.md` briefly.
- Mention adversarial evaluation script: `run_part_e_summary.py`.

## 1:55 - 2:00 (Close)
- "This submission includes code, UI, architecture, manual experiment logs, and evaluation tooling."
