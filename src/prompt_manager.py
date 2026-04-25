from typing import List, Tuple, Dict

class PromptManager:
    def __init__(self):
        # No tiktoken dependency — use lightweight char-count estimator.
        # Rule of thumb: 1 token ≈ 4 characters (accurate within ~10% for English).
        self.max_tokens = 2048
        self.min_similarity = 0.15

    def _count_tokens(self, text: str) -> int:
        """Estimate token count without tiktoken (1 token ≈ 4 chars)."""
        return max(1, len(text) // 4)
    
    def _manage_context_window(self, contexts: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
        """
        Rank, filter, and diversify contexts before prompt construction.
        """
        # Rank by similarity score
        ranked = sorted(contexts, key=lambda x: x[1], reverse=True)
        # Filter weak matches
        filtered = [c for c in ranked if c[1] >= self.min_similarity]
        if not filtered:
            filtered = ranked[:3]
        
        # Lightweight diversity by source (avoid many near-duplicate chunks)
        selected = []
        seen_sources = set()
        for doc, score in filtered:
            source = doc.get("source", "unknown")
            if source not in seen_sources or len(selected) < 2:
                selected.append((doc, score))
                seen_sources.add(source)
            if len(selected) >= 6:
                break
        return selected
        
    def construct_prompt(self, query: str, contexts: List[Tuple[Dict, float]], 
                        handle_hallucination: bool = True) -> Dict:
        """
        Manual prompt construction with hallucination control
        """
        contexts = self._manage_context_window(contexts)
        
        # Creative but simple: evidence-ledger response style
        system_prompt = """You are an AI assistant for Academic City University.
Use ONLY the retrieved context blocks below.
If evidence is insufficient, respond exactly:
'I don't have enough information to answer this question.'
Never invent facts."""
        
        if handle_hallucination:
            system_prompt += """
Output format:
1) Answer: concise answer.
2) Evidence: bullet points with [Source] tags.
3) Confidence: High/Medium/Low.
If confidence is Low, explain what is missing."""
        
        # Build context string with similarity scores for transparency
        context_str = ""
        total_tokens = self._count_tokens(system_prompt + query)

        for i, (doc, score) in enumerate(contexts):
            doc_text = f"\n[Document {i+1} | Source: {doc['source']} | Relevance: {score:.2f}]\n{doc['content']}"
            doc_tokens = self._count_tokens(doc_text)

            if total_tokens + doc_tokens > self.max_tokens - 200:  # Buffer for response
                context_str += "\n[Additional relevant documents omitted due to length constraints]"
                break

            context_str += doc_text
            total_tokens += doc_tokens
        
        final_prompt = f"""{system_prompt}

Context:{context_str}

User Question: {query}

Use only the context. If sources conflict, state the conflict."""
        
        return {
            "system_prompt": system_prompt,
            "context": context_str,
            "user_query": query,
            "full_prompt": final_prompt,
            "token_count": total_tokens,
            "retrieved_sources": [doc['source'] for doc, _ in contexts],
            "contexts_used": len(contexts)
        }
    
    def experiment_prompts(self, query: str, contexts: List[Tuple[Dict, float]]) -> List[Dict]:
        """
        Generate different prompt variations for experimentation (Part C)
        """
        variations = []
        
        # Variation 1: Strict factual
        strict = self.construct_prompt(query, contexts, handle_hallucination=True)
        strict['full_prompt'] = strict['full_prompt'].replace(
            "Use only the context", 
            "Provide a concise, factual answer in 2-3 sentences using only the context"
        )
        variations.append({"type": "strict", **strict})
        
        # Variation 2: Detailed analytical
        detailed = self.construct_prompt(query, contexts, handle_hallucination=True)
        detailed['full_prompt'] += "\nProvide a detailed analysis including specific figures and statistics."
        variations.append({"type": "detailed", **detailed})
        
        # Variation 3: No citation control (baseline for hallucination testing)
        baseline = self.construct_prompt(query, contexts, handle_hallucination=False)
        variations.append({"type": "baseline", **baseline})
        
        return variations