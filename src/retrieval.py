from sentence_transformers import CrossEncoder
import nltk
from nltk.corpus import wordnet
from typing import List, Tuple, Dict
import re
from src.embeddings import CustomEmbeddingPipeline

class AdvancedRetriever(CustomEmbeddingPipeline):
    def __init__(self):
        super().__init__()
        self.reranker = CrossEncoder(self.config.RERANKER_MODEL)
        self.min_vector_score = 0.2
        self.stopwords = {
            "the", "a", "an", "and", "or", "to", "for", "of", "in", "on",
            "at", "by", "with", "is", "are", "was", "were", "what", "who",
            "how", "when", "where", "why"
        }
        # Download wordnet for query expansion
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def expand_query(self, query: str) -> List[str]:
        """
        Query expansion using synonym substitution (Part B extension)
        Generates variations to improve recall
        """
        words = query.split()
        expanded = [query]  # Original query
        
        # Simple expansion: Add synonyms for key terms
        for word in words:
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != word:
                        synonyms.add(synonym)
            
            if synonyms:
                # Create query variations with synonyms
                for syn in list(synonyms)[:2]:  # Limit to 2 synonyms
                    new_query = query.replace(word, syn)
                    expanded.append(new_query)
        
        return list(set(expanded))[:4]  # Max 4 variations
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())
    
    def _keyword_score(self, query: str, content: str, use_fix: bool = True) -> float:
        if not use_fix:
            keywords = query.lower().split()
            if not keywords:
                return 0.0
            return sum(1 for kw in keywords if kw in content.lower()) / len(keywords)
        
        # Fix: exact-token keyword matching + stopword filtering
        query_tokens = [t for t in self._tokenize(query) if t not in self.stopwords]
        if not query_tokens:
            return 0.0
        
        content_tokens = set(self._tokenize(content))
        matched = sum(1 for token in query_tokens if token in content_tokens)
        return matched / len(query_tokens)
    
    def hybrid_search(self, query: str, k: int = None, use_fix: bool = True) -> List[Tuple[Dict, float]]:
        """
        Hybrid search: Keyword matching + Vector similarity
        """
        k = k or self.config.TOP_K
        
        # Vector search
        vector_results = self.similarity_search(query, k=k*2)
        
        # Keyword scoring (simple BM25 approximation)
        keyword_scores = []
        
        for chunk in self.chunks:
            content = chunk['content']
            score = self._keyword_score(query, content, use_fix=use_fix)
            keyword_scores.append(score)
        
        # Normalize and combine scores
        vector_dict = {id(item[0]): item[1] for item in vector_results}
        
        combined = []
        for i, chunk in enumerate(self.chunks):
            v_score = vector_dict.get(id(chunk), 0.0)
            k_score = keyword_scores[i]
            # Weighted combination
            final_score = 0.7 * v_score + 0.3 * k_score
            if not use_fix:
                combined.append((chunk, final_score))
            else:
                # Fix: filter weak semantic matches to reduce irrelevant hits
                if v_score >= self.min_vector_score or k_score >= 0.6:
                    combined.append((chunk, final_score))
        
        # Sort and return top k
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:k]
    
    def rerank_results(self, query: str, results: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
        """
        Cross-encoder re-ranking for better precision (Part B)
        """
        pairs = [[query, res[0]['content']] for res in results]
        scores = self.reranker.predict(pairs)
        
        reranked = [(results[i][0], float(scores[i])) for i in range(len(results))]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:self.config.RERANK_TOP_K]
    
    def retrieve(self, query: str, use_expansion: bool = False, use_reranking: bool = True) -> Dict:
        """
        Full retrieval pipeline with logging
        """
        log_entry = {
            "original_query": query,
            "stages": []
        }
        
        # Stage 1: Query expansion (optional)
        if use_expansion:
            queries = self.expand_query(query)
            all_results = []
            for q in queries:
                results = self.similarity_search(q, k=3)
                all_results.extend(results)
            # Deduplicate and sort
            seen = set()
            unique_results = []
            for res in all_results:
                if res[0]['content'] not in seen:
                    unique_results.append(res)
                    seen.add(res[0]['content'])
            results = unique_results[:self.config.TOP_K]
            log_entry["stages"].append({"expansion": queries})
        else:
            results = self.hybrid_search(query)
            log_entry["stages"].append({"method": "hybrid_search"})
        
        # Stage 2: Re-ranking
        if use_reranking:
            results = self.rerank_results(query, results)
            log_entry["stages"].append({"reranking": True})
        
        log_entry["final_results"] = [
            {"content": r[0]['content'][:100], "score": r[1]} 
            for r in results
        ]
        
        return {
            "results": results,
            "log": log_entry
        }
    
    def diagnose_failure_cases(self) -> List[Dict]:
        """
        Part B critical task:
        Show failure cases from old hybrid scoring and compare with fixed scoring.
        """
        synthetic_chunks = [
            "Region: Greater Accra. Party: NPP. Votes: 120000.",
            "Macroeconomic stability and smaller fiscal deficits supported recovery.",
            "This text is generic and includes filler words like is, in, the, and to."
        ]
        failure_queries = [
            "is in the to",
            "the and is",
            "what is in the document"
        ]
        report = []
        
        for q in failure_queries:
            before_scores = [self._keyword_score(q, c, use_fix=False) for c in synthetic_chunks]
            after_scores = [self._keyword_score(q, c, use_fix=True) for c in synthetic_chunks]
            report.append({
                "query": q,
                "before_fix_max_keyword_score": max(before_scores),
                "after_fix_max_keyword_score": max(after_scores),
                "before_fix_scores": before_scores,
                "after_fix_scores": after_scores
            })
        return report