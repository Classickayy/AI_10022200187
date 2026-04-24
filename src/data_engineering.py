import pandas as pd
import re
import pdfplumber
import os
from typing import List, Dict, Tuple
import json
from config import Config

class DataProcessor:
    def __init__(self):
        self.config = Config()
    
    def _resolve_data_path(self, configured_path: str, fallback_names: List[str]) -> str:
        """
        Resolve a configured path first, then try project-root fallback files.
        This keeps Part A runnable even when raw files were placed in root.
        """
        if os.path.exists(configured_path):
            return configured_path
        for name in fallback_names:
            if os.path.exists(name):
                return name
        raise FileNotFoundError(f"Could not locate data file. Tried: {configured_path}, {fallback_names}")
        
    def clean_text(self, text: str) -> str:
        """Manual text cleaning without external preprocessing libraries"""
        # Normalize whitespace and remove noisy control symbols
        text = re.sub(r'[\r\t]+', ' ', str(text))
        text = re.sub(r'\s+', ' ', text)
        # Keep punctuation useful for sentence boundaries
        text = re.sub(r'[^\w\s.,;:!?%-]', '', text)
        return text.strip()
    
    def load_csv(self) -> List[Dict]:
        """Load and clean election data"""
        csv_path = self._resolve_data_path(
            self.config.CSV_PATH,
            ["Ghana_Election_Result.csv"]
        )
        df = pd.read_csv(csv_path)
        
        # Data cleaning
        df.columns = [str(c).strip() for c in df.columns]
        if 'Region' not in df.columns:
            if 'New Region' in df.columns:
                df['Region'] = df['New Region']
            elif 'Old Region' in df.columns:
                df['Region'] = df['Old Region']
        if 'Party' not in df.columns and 'Code' in df.columns:
            # In this dataset, 'Code' is the per-region ballot / party line tag (NPP, NDC, OTHERS, etc.)
            df['Party'] = df['Code']
        for col in ['Region', 'Party', 'Candidate', 'Code']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('\u00a0', ' ').str.strip()
        if not {'Region', 'Party'}.issubset(set(df.columns)):
            raise KeyError("CSV is missing required columns after normalization. Expected at least Region+Party (or mappable New Region/Code).")
        df = df.dropna(subset=['Region', 'Party'])
        if 'Votes' in df.columns:
            df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
        else:
            raise KeyError("CSV is missing a 'Votes' column.")
        df = df.dropna(subset=['Votes'])
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df.drop_duplicates()
        
        # Convert rows to structured text documents
        documents = []
        for _, row in df.iterrows():
            text = f"Region: {row['Region']}. Party: {row['Party']}. Votes: {row['Votes']}. "
            text += f"Candidate: {row.get('Candidate', 'N/A')}. Year: {row.get('Year', 'N/A')}"
            documents.append({
                'content': self.clean_text(text),
                'source': 'election_data',
                'metadata': row.to_dict()
            })
        return documents
    
    def load_pdf(self) -> List[Dict]:
        """Load and clean budget PDF with semantic chunking"""
        documents = []
        pdf_path = self._resolve_data_path(
            self.config.PDF_PATH,
            ["2025-Budget-Statement-and-Economic-Policy_v4.pdf"]
        )
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    full_text += f"\n\n[Page {page_num}]\n{text}"
            
            # Semantic chunking: Split by headers/sections
            sections = re.split(r'\n(?=[A-Z][A-Z\s]{3,}\n)', full_text)
            
            for i, section in enumerate(sections):
                if len(section.strip()) > 50:  # Filter out headers
                    documents.append({
                        'content': self.clean_text(section),
                        'source': 'budget_2025',
                        'metadata': {
                            'section_id': i,
                            'type': 'budget_statement'
                        }
                    })
        return documents
    
    def fixed_word_chunking(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Strategy 1: Fixed-size word chunking.
        Baseline method for comparing against recursive sentence-aware chunking.
        """
        chunk_size = chunk_size or self.config.CHUNK_SIZE
        overlap = overlap if overlap is not None else self.config.CHUNK_OVERLAP
        words = text.split()
        if not words:
            return []
        
        chunks = []
        step = max(1, chunk_size - overlap)
        for start in range(0, len(words), step):
            chunk_words = words[start:start + chunk_size]
            if chunk_words:
                chunks.append(" ".join(chunk_words))
            if start + chunk_size >= len(words):
                break
        return chunks
    
    def recursive_chunking(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Strategy 2: Sentence-aware recursive chunking (Part A implementation)
        Justification: Recursive splitting preserves sentence boundaries better than fixed-size
        """
        chunk_size = chunk_size or self.config.CHUNK_SIZE
        overlap = overlap if overlap is not None else self.config.CHUNK_OVERLAP
        
        # Split into sentences first (rough approximation)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > chunk_size:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                # Keep overlap sentences
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    def _chunk_document(self, text: str, strategy: str) -> List[str]:
        if strategy == "fixed":
            return self.fixed_word_chunking(text)
        return self.recursive_chunking(text)
    
    def _evaluate_chunk_retrieval_quality(self, chunks: List[str], eval_queries: List[Tuple[str, List[str]]]) -> Dict:
        """
        Simple retrieval-quality proxy for Part A:
        - hit_rate: at least one chunk contains all expected keywords
        - avg_keyword_coverage: best overlap ratio per query
        """
        hits = 0
        coverages = []
        
        for _, keywords in eval_queries:
            best_coverage = 0.0
            for chunk in chunks:
                chunk_lower = chunk.lower()
                matched = sum(1 for kw in keywords if kw in chunk_lower)
                coverage = matched / len(keywords)
                best_coverage = max(best_coverage, coverage)
            if best_coverage >= 1.0:
                hits += 1
            coverages.append(best_coverage)
        
        total = len(eval_queries) if eval_queries else 1
        return {
            "hit_rate": hits / total,
            "avg_keyword_coverage": sum(coverages) / total
        }
    
    def compare_chunking_strategies(self) -> Dict:
        """Comparative analysis of chunking impact on retrieval quality."""
        docs = self.load_csv() + self.load_pdf()
        doc_texts = [d["content"] for d in docs]
        
        fixed_chunks = []
        recursive_chunks = []
        for text in doc_texts:
            fixed_chunks.extend(self.fixed_word_chunking(text))
            recursive_chunks.extend(self.recursive_chunking(text))
        
        # Keyword triplets are chosen from the actual text so this comparison is not "empty" on the real data.
        eval_queries = [
            ("Greater Accra election row contains region + party + vote signal", ["greater", "accra", "votes"]),
            ("Budget text contains education and revenue themes", ["education", "revenue", "thematic"]),
            ("Budget text contains fiscal deficit phrasing", ["fiscal", "deficits", "macroeconomic"])
        ]
        
        fixed_quality = self._evaluate_chunk_retrieval_quality(fixed_chunks, eval_queries)
        recursive_quality = self._evaluate_chunk_retrieval_quality(recursive_chunks, eval_queries)
        
        return {
            "chunk_params": {
                "chunk_size": self.config.CHUNK_SIZE,
                "chunk_overlap": self.config.CHUNK_OVERLAP
            },
            "fixed": {
                "num_chunks": len(fixed_chunks),
                **fixed_quality
            },
            "recursive": {
                "num_chunks": len(recursive_chunks),
                **recursive_quality
            }
        }
    
    def process_all(self, strategy: str = "recursive"):
        """Execute full pipeline"""
        all_chunks = []
        
        # Process CSV
        csv_docs = self.load_csv()
        for doc in csv_docs:
            # CSV entries are already small, may not need chunking
            if len(doc['content'].split()) > self.config.CHUNK_SIZE:
                chunks = self._chunk_document(doc['content'], strategy)
                for chunk in chunks:
                    all_chunks.append({**doc, 'content': chunk, 'chunking_strategy': strategy})
            else:
                all_chunks.append({**doc, 'chunking_strategy': "row_level"})
        
        # Process PDF
        pdf_docs = self.load_pdf()
        for doc in pdf_docs:
            chunks = self._chunk_document(doc['content'], strategy)
            for chunk in chunks:
                all_chunks.append({**doc, 'content': chunk, 'chunking_strategy': strategy})
        
        # Save processed data
        os.makedirs(self.config.PROCESSED_DIR, exist_ok=True)
        with open(f"{self.config.PROCESSED_DIR}/chunks.json", 'w') as f:
            json.dump(all_chunks, f, indent=2)
            
        return all_chunks