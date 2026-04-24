import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_pipeline import RAGPipeline
from src.feedback_memory import FeedbackMemoryRAG
from config import Config

st.set_page_config(page_title="Academic City RAG Assistant", layout="wide")

# Initialize
@st.cache_resource
def load_pipeline():
    pipe = RAGPipeline()
    return pipe, FeedbackMemoryRAG(pipe)

pipeline, memory = load_pipeline()
pipeline.set_feedback_memory(memory)

# Show banner if no LLM provider is configured (retrieval-only mode)
if not hasattr(pipeline, "llm_provider") or pipeline.llm_provider is None:
    st.warning(
        "No LLM provider configured. The app will run in retrieval-only mode. "
        "Set the OPENAI_API_KEY or GROQ_API_KEY environment variable to enable LLM functionality."
    )
elif pipeline.llm_provider == "groq" and getattr(pipeline, "groq_client", None) is None:
    st.warning(
        "Groq provider selected but GROQ_API_KEY is not set. Set GROQ_API_KEY to enable Groq LLM calls."
    )

st.title("🏛️ Ghana Facts AI Chatbot")


# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    use_expansion = st.checkbox("Use Query Expansion", value=False)
    use_reranking = st.checkbox("Use Re-ranking", value=True)
    prompt_type = st.selectbox("Prompt Type", ["strict", "detailed", "baseline"])
    
    st.markdown("---")
    

# Main interface
query = st.text_input("Ask about Ghana Elections or 2025 Budget:", 
                      placeholder="e.g., What was the budget for education?")

if query:
    # If no LLM provider or Groq client missing, run retrieval-only flow and avoid LLM calls
    if not hasattr(pipeline, "llm_provider") or pipeline.llm_provider is None or \
       (pipeline.llm_provider == "groq" and getattr(pipeline, "groq_client", None) is None):
        with st.spinner("Retrieving documents (LLM disabled)..."):
            retrieval_output = pipeline.retriever.retrieve(
                query,
                use_expansion=use_expansion,
                use_reranking=use_reranking
            )

        st.subheader("🔍 Retrieved Context")
        for i, (doc, score) in enumerate(retrieval_output['results']):
            with st.expander(f"Document {i+1} | {doc['source']} | Score: {score:.3f}"):
                st.write(doc['content'])

        st.info("LLM generation is disabled because no LLM provider is configured. "
                "Set the OPENAI_API_KEY or GROQ_API_KEY environment variable to enable model responses.")

    else:
        with st.spinner("Processing..."):
            # Run pipeline (retrieval + LLM)
            result = pipeline.run(
                query,
                use_expansion=use_expansion,
                use_reranking=use_reranking,
                prompt_type=prompt_type
            )

            # Display retrieved chunks (Part of requirements)
            st.subheader("🔍 Retrieved Context")
            for i, doc in enumerate(result['retrieved_documents']):
                with st.expander(f"Document {i+1} | {doc['source']} | Score: {doc['similarity_score']:.3f}"):
                    st.write(doc['content'])

            # Display final prompt (for transparency/debugging)
            with st.expander("📝 Final Prompt Sent to LLM"):
                st.text(result['final_prompt_sent_to_llm'])

            # Display response
            st.subheader("💬 Response")
            st.write(result['response'])

            # Performance metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Retrieval Time", f"{result['performance']['retrieval_time']:.2f}s")
            col2.metric("Generation Time", f"{result['performance']['generation_time']:.2f}s")
            col3.metric("Total Time", f"{result['performance']['total_time']:.2f}s")

            # Feedback mechanism (Part G)
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Good Response"):
                    memory.store_interaction(query, result['response'], 
                                               result['retrieved_documents'], "positive")
                    st.success("Thank you for your feedback!")
            with col2:
                if st.button("👎 Bad Response"):
                    memory.store_interaction(query, result['response'], 
                                               result['retrieved_documents'], "negative")
                    st.error("Thank you for your feedback!")

# Documentation section
st.markdown("---")
st.markdown("### 📚 System Documentation")
st.markdown("""
**Architecture:**
- **Data Sources**: Ghana Election Results CSV + 2025 Budget PDF
- **Chunking**: Recursive character splitting (512 tokens, 50 overlap)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS (IndexFlatIP)
- **Retrieval**: Hybrid (Keyword + Vector) with optional Query Expansion and Cross-Encoder Re-ranking
- **LLM**: GPT-3.5-Turbo with custom prompt engineering
- **Innovation**: Feedback loop for continuous retrieval improvement
""")