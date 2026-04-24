# CS4241 RAG Examination Project

This repository contains a complete Retrieval-Augmented Generation (RAG) system for question answering over:
- Ghana election results (CSV)
- Ghana 2025 budget statement (PDF)

The system is implemented with custom data processing, FAISS vector retrieval, prompt engineering, adversarial evaluation, and a feedback-memory innovation loop.

## 1) Application Deliverable

### GitHub Solution
- Repository contains source code, logs, architecture documentation, and experiment scripts.

### Simple UI
- Framework: `Streamlit`
- Entry point: `app.py`

### Required Features in UI
- Query input: `st.text_input(...)`
- Display retrieved chunks: expandable context cards with source + score
- Show final response: rendered LLM response panel
- Also included:
  - Similarity score display
  - Final prompt preview
  - Performance timings

## 2) Project Structure

- `app.py` - Streamlit interface
- `config.py` - configuration (models, chunking, paths, API config)
- `src/data_engineering.py` - cleaning + chunking + chunk strategy comparison
- `src/embeddings.py` - embedding pipeline + FAISS index
- `src/retrieval.py` - top-k retrieval, hybrid scoring, expansion/reranking, failure diagnosis
- `src/prompt_manager.py` - context window management + hallucination control prompts
- `src/rag_pipeline.py` - full pipeline orchestration + stage logging
- `src/evaluator.py` - adversarial testing + RAG vs pure LLM evaluation
- `src/feedback_memory.py` - innovation: feedback-driven retrieval enhancement
- `ARCHITECTURE.md` - architecture diagram + explanation
- `run_part_e_summary.py` - one-command Part E summary script
- `logs/` - pipeline and evaluation outputs

## 3) Setup and Run

### Install
```bash
pip install -r requirements.txt
```

### Run application
```bash
streamlit run app.py
```

### Environment variables
- `OPENAI_API_KEY` (if using OpenAI)
- or `GROK_API_KEY` + `GROK_API_URL` (if using Grok endpoint)
- optional: `LLM_PROVIDER` = `openai` or `grok`

## 4) Manual Experiment Artifacts

- Main evidence log: `MANUAL_EXPERIMENT_LOG.md`
- Part E rerun helper: `run_part_e_summary.py`
- Saved evaluation JSON: `logs/adversarial_evaluation.json`
- Pipeline stage logs: `logs/pipeline_YYYYMMDD.jsonl`

## 5) Documentation Deliverables

- Detailed system/design documentation: this `README.md`
- Architecture/design rationale: `ARCHITECTURE.md`
- Final submission checklist: `FINAL_DELIVERABLES_CHECKLIST.md`
- Video walkthrough guide: `VIDEO_WALKTHROUGH_GUIDE.md`

## 6) Notes for Examiner

- If LLM API is not reachable, retrieval and pipeline stages still run and are logged.
- The app has a retrieval-only fallback mode when no LLM provider is configured.
- For clean Part E metrics, ensure API endpoint is valid before rerunning:
```bash
python run_part_e_summary.py
```
