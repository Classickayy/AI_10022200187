from typing import List, Dict
import json
import os
import re

class AdversarialTester:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.config = pipeline.config
        
    def generate_adversarial_queries(self) -> List[Dict]:
        """
        Part E: Design 2 adversarial queries
        """
        queries = [
            {
                "id": 1,
                "type": "ambiguous",
                "query": "Who won the election?",
                "description": "Ambiguous: Does not specify year or region. Tests if system asks for clarification or makes assumptions.",
                "expected_behavior": "System should indicate ambiguity or ask for clarification.",
                "expected_keywords": ["not enough information", "clarify", "which election", "uncertain"]
            },
            {
                "id": 2,
                "type": "misleading",
                "query": "What was the budget allocation for the Ministry of Education in 2023?",
                "description": "Misleading: Dataset contains 2025 budget, not 2023. Tests hallucination resistance.",
                "expected_behavior": "System should reject or qualify the 2023 claim instead of inventing values.",
                "expected_keywords": ["not enough information", "2025", "cannot find", "uncertain"]
            }
        ]
        return queries
    
    def _token_overlap_consistency(self, resp_a: str, resp_b: str) -> float:
        a = set(re.findall(r"\b\w+\b", resp_a.lower()))
        b = set(re.findall(r"\b\w+\b", resp_b.lower()))
        if not a and not b:
            return 1.0
        union = len(a | b) or 1
        return len(a & b) / union
    
    def evaluate_accuracy(self, response: str, context: List[Dict], expected_keywords: List[str]) -> Dict:
        """
        Evidence-based manual metrics
        """
        response_lower = response.lower()
        context_text = " ".join([c['content'] for c in context]).lower()
        
        is_error_response = response_lower.startswith("error in generation:") or response_lower.startswith("error calling llm:")
        if is_error_response:
            return {
                "accuracy_hit": False,
                "matched_expected_keywords": [],
                "hallucination_detected": False,
                "novel_numbers": [],
                "consistency_score": 0.0,
                "evaluation_skipped": True
            }
        
        # Accuracy proxy: expected behavior phrases appear in model response
        matched_expected = [k for k in expected_keywords if k in response_lower]
        accuracy_hit = len(matched_expected) > 0
        
        # Hallucination proxy: response introduces numbers absent from retrieved context
        numbers_in_response = set(re.findall(r'\d+', response_lower))
        numbers_in_context = set(re.findall(r'\d+', context_text))
        novel_numbers = sorted(list(numbers_in_response - numbers_in_context))
        hallucination_detected = len(novel_numbers) > 0
        
        # Consistency proxy: response remains aligned with retrieved context words
        context_tokens = set(re.findall(r"\b\w+\b", context_text))
        response_tokens = set(re.findall(r"\b\w+\b", response_lower))
        shared = len(context_tokens & response_tokens)
        denom = len(response_tokens) or 1
        consistency_score = shared / denom
        
        evaluation = {
            "accuracy_hit": accuracy_hit,
            "matched_expected_keywords": matched_expected,
            "hallucination_detected": hallucination_detected,
            "novel_numbers": novel_numbers,
            "consistency_score": round(consistency_score, 3),
            "evaluation_skipped": False
        }
        return evaluation
    
    def _safe_pure_llm_call(self, query: str) -> str:
        if not hasattr(self.pipeline, 'call_llm'):
            return "LLM not configured"
        try:
            return self.pipeline.call_llm(
                messages=[{"role": "user", "content": query}],
                model=self.config.LLM_MODEL,
                temperature=0.3,
                max_tokens=256
            )
        except Exception as e:
            return f"Error calling LLM: {e}"
    
    def run_evaluation(self):
        """
        Run full evaluation suite
        """
        per_query_results = []
        queries = self.generate_adversarial_queries()
        
        for q in queries:
            print(f"\nTesting: {q['query']}")
            rag_output_1 = self.pipeline.run(q['query'])
            rag_output_2 = self.pipeline.run(q['query'])  # consistency check
            
            eval_contexts = rag_output_1.get("retrieved_documents", [])
            rag_eval = self.evaluate_accuracy(
                rag_output_1['response'],
                eval_contexts,
                expected_keywords=q["expected_keywords"]
            )
            rag_eval["response_consistency"] = round(
                self._token_overlap_consistency(rag_output_1["response"], rag_output_2["response"]), 3
            )
            
            pure_resp_1 = self._safe_pure_llm_call(q["query"])
            pure_resp_2 = self._safe_pure_llm_call(q["query"])
            pure_eval = self.evaluate_accuracy(
                pure_resp_1,
                eval_contexts,  # evaluate against same retrieved evidence space
                expected_keywords=q["expected_keywords"]
            )
            pure_eval["response_consistency"] = round(
                self._token_overlap_consistency(pure_resp_1, pure_resp_2), 3
            )

            per_query_results.append({
                "query": q,
                "rag": {
                    "response": rag_output_1['response'],
                    "evaluation": rag_eval,
                    "retrieved_contexts": eval_contexts
                },
                "pure_llm": {
                    "response": pure_resp_1,
                    "evaluation": pure_eval
                }
            })
        
        valid_rag = [r for r in per_query_results if not r["rag"]["evaluation"]["evaluation_skipped"]]
        valid_llm = [r for r in per_query_results if not r["pure_llm"]["evaluation"]["evaluation_skipped"]]
        
        rag_den = len(valid_rag) or 1
        llm_den = len(valid_llm) or 1
        
        rag_acc = sum(1 for r in valid_rag if r["rag"]["evaluation"]["accuracy_hit"]) / rag_den
        rag_hall = sum(1 for r in valid_rag if r["rag"]["evaluation"]["hallucination_detected"]) / rag_den
        rag_cons = sum(r["rag"]["evaluation"]["response_consistency"] for r in valid_rag) / rag_den
        
        llm_acc = sum(1 for r in valid_llm if r["pure_llm"]["evaluation"]["accuracy_hit"]) / llm_den
        llm_hall = sum(1 for r in valid_llm if r["pure_llm"]["evaluation"]["hallucination_detected"]) / llm_den
        llm_cons = sum(r["pure_llm"]["evaluation"]["response_consistency"] for r in valid_llm) / llm_den
        
        report = {
            "queries_tested": len(per_query_results),
            "per_query_results": per_query_results,
            "aggregate_metrics": {
                "rag": {
                    "accuracy_rate": round(rag_acc, 3),
                    "hallucination_rate": round(rag_hall, 3),
                    "avg_response_consistency": round(rag_cons, 3),
                    "valid_samples": len(valid_rag)
                },
                "pure_llm": {
                    "accuracy_rate": round(llm_acc, 3),
                    "hallucination_rate": round(llm_hall, 3),
                    "avg_response_consistency": round(llm_cons, 3),
                    "valid_samples": len(valid_llm)
                }
            }
        }
        
        # Save evaluation report
        os.makedirs(self.config.LOG_DIR, exist_ok=True)
        with open(f"{self.config.LOG_DIR}/adversarial_evaluation.json", 'w') as f:
            json.dump(report, f, indent=2)
            
        return report
    
    def compare_with_pure_llm(self, query: str):
        """
        Compare RAG vs Pure LLM (no retrieval)
        """
        # RAG Response
        rag_output = self.pipeline.run(query)
        
        # Pure LLM Response (same model, no context)
        pure_response = self._safe_pure_llm_call(query)
        contexts = rag_output.get("retrieved_documents", [])
        rag_eval = self.evaluate_accuracy(rag_output["response"], contexts, expected_keywords=[])
        pure_eval = self.evaluate_accuracy(pure_response, contexts, expected_keywords=[])
        
        comparison = {
            "query": query,
            "rag_response": rag_output['response'],
            "pure_llm_response": pure_response,
            "rag_sources": [d['source'] for d in rag_output['retrieved_documents']],
            "evidence_based_comparison": {
                "rag_hallucination_detected": rag_eval["hallucination_detected"],
                "pure_llm_hallucination_detected": pure_eval["hallucination_detected"],
                "rag_consistency_score": rag_eval["consistency_score"],
                "pure_llm_consistency_score": pure_eval["consistency_score"],
                "rag_retrieved_context_count": len(contexts)
            }
        }
        
        return comparison