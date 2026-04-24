import json
import time
import os
from datetime import datetime
from typing import Dict, List
from config import Config
from src.retrieval import AdvancedRetriever
from src.prompt_manager import PromptManager
from openai import OpenAI
from groq import Groq

class RAGPipeline:
    def __init__(self):
        self.config = Config()
        self.retriever = AdvancedRetriever()
        self.prompt_manager = PromptManager()
        self.retriever.load_index()

        # Configure LLM provider (openai or groq)
        openai_key = self.config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        groq_key = getattr(self.config, "GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY")
        # Fallback to legacy GROK env vars for backward compatibility
        if not groq_key:
            groq_key = getattr(self.config, "GROK_API_KEY", None) or os.getenv("GROK_API_KEY")

        # Provider priority: explicit LLM_PROVIDER env/config, else prefer OpenAI if key present
        provider = os.getenv("LLM_PROVIDER") or getattr(self.config, "LLM_PROVIDER", None)
        if not provider:
            if groq_key and not openai_key:
                provider = "groq"
            elif openai_key:
                provider = "openai"
            else:
                provider = None

        self.llm_provider = provider
        self.openai_client = None
        self.groq_client = None
        self.groq_key = None

        if provider == "openai" and openai_key:
            self.openai_client = OpenAI(api_key=openai_key)
        elif provider == "groq" and groq_key:
            self.groq_client = Groq(api_key=groq_key)
            self.groq_key = groq_key
        else:
            if provider:
                print(f"Warning: LLM provider '{provider}' configured but no API key found.")
        
        # Optional innovation hook: feedback memory can be attached from app layer
        self.feedback_memory = None
    
    def set_feedback_memory(self, feedback_memory):
        self.feedback_memory = feedback_memory
        
    def log_stage(self, stage_name: str, data: dict):
        """Implement logging at each stage"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage_name,
            "data": data
        }
        
        # Append to daily log file
        log_file = f"{self.config.LOG_DIR}/pipeline_{datetime.now().strftime('%Y%m%d')}.jsonl"
        os.makedirs(self.config.LOG_DIR, exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
        return log_entry

    def call_llm(self, messages: List[Dict], model: str = None, temperature: float = 0.3,
                 max_tokens: int = 500) -> str:
        """Unified LLM caller. Supports OpenAI and Groq.

        For Groq, set `GROQ_API_KEY` environment variable (or in Config).
        """
        if self.llm_provider == "openai":
            if not self.openai_client:
                raise RuntimeError("OpenAI client not configured")
            resp = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return resp.choices[0].message.content

        if self.llm_provider == "groq":
            return self._call_groq(messages, model=model, temperature=temperature, max_tokens=max_tokens)

        raise RuntimeError("No LLM provider configured (set OPENAI_API_KEY or GROQ_API_KEY)")

    def _call_groq(self, messages: List[Dict], model: str = None, temperature: float = 0.3,
                   max_tokens: int = 500) -> str:
        """Call Groq API using official client."""
        if not self.groq_client:
            raise RuntimeError("Groq client not configured")
        
        try:
            response = self.groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Groq API call failed: {e}")
    

    
    def run(self, query: str, use_expansion: bool = False, use_reranking: bool = True,
            prompt_type: str = "strict") -> Dict:
        """
        Complete pipeline: Query -> Retrieval -> Prompt -> LLM -> Response
        """
        start_time = time.time()
        # Stage 1: Retrieval
        retrieval_start = time.time()
        retrieval_output = self.retriever.retrieve(
            query,
            use_expansion=use_expansion,
            use_reranking=use_reranking
        )
        retrieval_time = time.time() - retrieval_start
        
        # Innovation component: feedback-driven retrieval refinement
        if self.feedback_memory is not None:
            before_scores = [score for _, score in retrieval_output.get("results", [])]
            retrieval_output["results"] = self.feedback_memory.enhance_retrieval(
                query, retrieval_output.get("results", [])
            )
            after_scores = [score for _, score in retrieval_output.get("results", [])]
            self.log_stage("feedback_memory_boost", {
                "applied": True,
                "before_top_score": before_scores[0] if before_scores else 0.0,
                "after_top_score": after_scores[0] if after_scores else 0.0
            })
        
        self.log_stage("retrieval", {
            "query": query,
            "num_results": len(retrieval_output['results']),
            "expansion_used": use_expansion,
            "reranking_used": use_reranking,
            "time_seconds": retrieval_time
        })
        
        # Stage 2: Context Selection & Prompt Construction
        contexts = retrieval_output['results']
        selected_contexts = contexts[:self.config.TOP_K]
        self.log_stage("context_selection", {
            "selected_count": len(selected_contexts),
            "sources": [doc.get("source", "unknown") for doc, _ in selected_contexts],
            "similarity_scores": [float(score) for _, score in selected_contexts]
        })
        
        if prompt_type == "strict":
            prompt_data = self.prompt_manager.construct_prompt(query, selected_contexts, handle_hallucination=True)
        elif prompt_type in ["detailed", "baseline"]:
            variations = self.prompt_manager.experiment_prompts(query, selected_contexts)
            selected = next((v for v in variations if v["type"] == prompt_type), variations[0])
            prompt_data = selected
        else:
            prompt_data = self.prompt_manager.construct_prompt(query, selected_contexts, handle_hallucination=True)
        
        self.log_stage("prompt_construction", {
            "token_count": prompt_data['token_count'],
            "prompt_preview": prompt_data['full_prompt'][:200]
        })
        
        # Stage 3: LLM Generation
        llm_start = time.time()

        try:
            messages = [
                {"role": "system", "content": prompt_data['system_prompt']},
                {"role": "user", "content": prompt_data['full_prompt']}
            ]

            generated_text = self.call_llm(messages=messages, model=self.config.LLM_MODEL,
                                           temperature=0.3, max_tokens=500)
            llm_time = time.time() - llm_start

            self.log_stage("generation", {
                "model": self.config.LLM_MODEL,
                "response_length": len(generated_text),
                "time_seconds": llm_time
            })

        except Exception as e:
            generated_text = f"Error in generation: {str(e)}"
            llm_time = 0
        
        total_time = time.time() - start_time
        
        # Compile final output with all intermediate stages visible
        output = {
            "query": query,
            "retrieved_documents": [
                {
                    "content": doc['content'],
                    "source": doc['source'],
                    "similarity_score": score
                } for doc, score in selected_contexts
            ],
            "similarity_scores": [score for _, score in selected_contexts],
            "final_prompt_sent_to_llm": prompt_data['full_prompt'],
            "token_count": prompt_data['token_count'],
            "response": generated_text,
            "sources_cited": prompt_data['retrieved_sources'],
            "performance": {
                "retrieval_time": retrieval_time,
                "generation_time": llm_time,
                "total_time": total_time
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "prompt_type": prompt_type,
                "expansion_used": use_expansion,
                "reranking_used": use_reranking
            },
            "pipeline_trace": {
                "retrieval_log": retrieval_output.get("log", {}),
                "context_selection_count": len(selected_contexts)
            }
        }
        
        self.log_stage("complete_pipeline", {"success": True})
        return output
    
    def run_prompt_experiment(self, query: str, use_expansion: bool = False) -> Dict:
        """
        Part C experiment:
        Run same query with strict, detailed, and baseline prompts and compare outputs.
        """
        retrieval_output = self.retriever.retrieve(query, use_expansion=use_expansion)
        contexts = retrieval_output["results"]
        variations = self.prompt_manager.experiment_prompts(query, contexts)
        
        experiment_rows = []
        for v in variations:
            messages = [
                {"role": "system", "content": v["system_prompt"]},
                {"role": "user", "content": v["full_prompt"]}
            ]
            prompt_lower = (v["system_prompt"] + "\n" + v["full_prompt"]).lower()
            prompt_signals = {
                "enforces_uncertainty_statement": "i don't have enough information to answer this question." in prompt_lower,
                "requires_evidence_section": "evidence:" in prompt_lower,
                "requires_confidence_label": "confidence:" in prompt_lower
            }
            try:
                response = self.call_llm(messages=messages, model=self.config.LLM_MODEL, temperature=0.3, max_tokens=500)
                generation_status = "ok"
            except Exception as e:
                response = f"Error in generation: {str(e)}"
                generation_status = "error"
            
            lower = response.lower()
            uncertainty_flag = any(p in lower for p in ["i don't have enough information", "uncertain", "not enough information"])
            citation_flag = "[" in response and "]" in response
            
            experiment_rows.append({
                "prompt_type": v["type"],
                "token_count": v["token_count"],
                "contexts_used": v.get("contexts_used", len(contexts)),
                "response": response,
                "response_length": len(response),
                "generation_status": generation_status,
                "signals": {
                    "has_uncertainty_control": uncertainty_flag,
                    "has_source_style_citation": citation_flag,
                    "prompt_signals": prompt_signals
                },
                "prompt_preview": v["full_prompt"][:220]
            })
        
        return {
            "query": query,
            "results": experiment_rows
        }