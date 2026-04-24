import json
import os
from datetime import datetime
from typing import List, Dict
import re

class FeedbackMemoryRAG:
    """
    Innovation Component: Continuous learning from user feedback
    Stores successful Q&A pairs to improve future retrieval
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.feedback_file = f"{self.pipeline.config.LOG_DIR}/feedback_memory.jsonl"
        self.qa_memory = []
        self.load_memory()
        
    def load_memory(self):
        """Load historical Q&A pairs"""
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'r') as f:
                for line in f:
                    self.qa_memory.append(json.loads(line))
    
    def store_interaction(self, query: str, response: str, contexts: List[Dict], 
                         feedback: str = None):
        """
        Store interaction for future retrieval improvement
        feedback: 'positive', 'negative', or None
        """
        # Support both formats: list of (doc, score) tuples or list of dicts
        contexts_preview = []
        if contexts:
            first = contexts[0]
            if isinstance(first, tuple) or isinstance(first, list):
                contexts_preview = [c.get('content', '')[:200] for c, _ in contexts]
            elif isinstance(first, dict):
                contexts_preview = [c.get('content', '')[:200] for c in contexts]
            else:
                contexts_preview = [str(c)[:200] for c in contexts]

        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "contexts": contexts_preview,
            "feedback": feedback,
            "embedding": None  # Could store embedding for similarity search
        }
        
        with open(self.feedback_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")
        
        self.qa_memory.append(entry)
    
    def enhance_retrieval(self, query: str, initial_results: List) -> List:
        """
        Check if similar queries were answered successfully before
        """
        if not self.qa_memory:
            return initial_results
        
        # Simple keyword matching for similar past queries
        query_keywords = set(re.findall(r"\b\w+\b", query.lower()))
        if not query_keywords:
            return initial_results
        
        best_matches = []
        
        for memory in self.qa_memory[-100:]:  # Check last 100 interactions
            if memory.get('feedback') == 'positive':
                mem_keywords = set(re.findall(r"\b\w+\b", memory['query'].lower()))
                overlap = len(query_keywords & mem_keywords)
                if overlap > len(query_keywords) * 0.5:  # 50% overlap
                    best_matches.append(memory)
        
        if not best_matches:
            return initial_results
        
        # Build memory token set from contexts of positive historical interactions
        memory_tokens = set()
        for m in best_matches:
            for ctx in m.get("contexts", []):
                memory_tokens.update(re.findall(r"\b\w+\b", str(ctx).lower()))
        
        # Support tuples from retriever: (doc, score)
        if initial_results and isinstance(initial_results[0], tuple):
            rescored = []
            for doc, score in initial_results:
                doc_tokens = set(re.findall(r"\b\w+\b", doc.get("content", "").lower()))
                overlap = len(doc_tokens & memory_tokens)
                bonus = min(0.08, overlap * 0.005)  # small bounded bonus
                rescored.append((doc, float(score) + bonus))
            rescored.sort(key=lambda x: x[1], reverse=True)
            return rescored
        
        # Fallback for dict-style results with similarity_score
        if initial_results and isinstance(initial_results[0], dict):
            rescored = []
            for item in initial_results:
                text = item.get("content", "")
                doc_tokens = set(re.findall(r"\b\w+\b", text.lower()))
                overlap = len(doc_tokens & memory_tokens)
                bonus = min(0.08, overlap * 0.005)
                new_item = dict(item)
                new_item["similarity_score"] = float(item.get("similarity_score", 0.0)) + bonus
                rescored.append(new_item)
            rescored.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
            return rescored
        
        return initial_results
    
    def get_conversation_memory(self, session_id: str) -> List[Dict]:
        """
        Memory-based RAG: Retrieve previous turns in conversation
        for multi-turn coherence
        """
        # Filter by session_id if implemented
        return self.qa_memory[-5:]  # Last 5 turns