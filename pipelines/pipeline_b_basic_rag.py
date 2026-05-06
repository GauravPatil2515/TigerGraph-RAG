"""
Module: pipeline_b_basic_rag.py
Description: Pipeline B — standard vector RAG using ChromaDB for similarity search. 
             Represents the current industry standard. Uses relevance threshold 
             filtering (distance < 1.2) to prevent context injection from 
             irrelevant documents.

Author: Gaurav Patil
Project: GraphRAG Inference Hackathon — TigerGraph 2026
GitHub: https://github.com/GauravPatil2515/TigerGraph-RAG

Key Features:
    - ChromaDB vector retrieval using Sentence Transformers
    - Relevance-based filtering of retrieved document snippets
    - Instruction-tuned prompt to minimize hallucinations
"""

import time
import logging
import os
import sys
from typing import Dict, Any
from groq import Groq

# Ensure project root is in path for config import
sys.path.insert(0, os.getcwd())
from config import GROQ_API_KEY, GROQ_MODEL, COST_PER_1K
from ingest.chroma_ingest import get_collection

# Configure structured logging
logger = logging.getLogger(__name__)

class BasicRAGPipeline:
    """
    BasicRAGPipeline: Pipeline B of the 3-pipeline benchmark system.

    Implements standard vector-based Retrieval Augmented Generation (RAG).
    Retrieves documents from the 'pubmedqa' ChromaDB collection, which
    contains medical QA pairs and contexts embedded with 'all-MiniLM-L6-v2'.

    Attributes:
        client (Groq): Initialized Groq LLM client
        name (str): Pipeline identifier for CSV logging
        collection: ChromaDB collection handle for 'pubmedqa'
    """
    
    def __init__(self) -> None:
        """Initialize the Basic RAG pipeline with Groq client and ChromaDB connection."""
        self.client = Groq(api_key=GROQ_API_KEY)
        self.name: str = "basic_rag"
        # Accessing the 'pubmedqa' collection containing 1,000 embedded medical records
        self.collection = get_collection()
        logger.info(f"✅ {self.name} pipeline initialized")

    def retrieve_context(self, query: str) -> str:
        """
        Retrieve relevant context from ChromaDB with relevance filtering.
        
        Performs a vector similarity search and filters results based on
        a distance threshold to ensure context quality.

        Args:
            query: User's medical question text.

        Returns:
            str: Concatenated text of relevant document snippets, or empty string.
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=3,
                include=["documents", "distances"]
            )
            
            if not results["documents"] or not results["documents"][0]:
                return ""

            docs = results["documents"][0]
            dists = results["distances"][0]
            
            # Empirically determined: cosine distances > 1.2 produce irrelevant
            # context that degrades answer quality more than no context at all.
            # Filtering out low-confidence matches to maintain medical precision.
            RELEVANCE_THRESHOLD: float = 1.2
            relevant = []
            for doc, dist in zip(docs, dists):
                if dist < RELEVANCE_THRESHOLD:
                    relevant.append(doc)
                else:
                    logger.debug(f"Skipping irrelevant snippet with distance {dist:.4f}")
            
            return "\n\n---\n\n".join(relevant[:2]) if relevant else ""
            
        except Exception as e:
            logger.error(f"ChromaDB retrieval error in {self.name}: {e}")
            return ""

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the Basic RAG pipeline for a given query.
        
        Retrieves context and then performs augmented LLM inference.

        Args:
            query: User's medical question text.

        Returns:
            dict: Standard pipeline result dictionary with answer and metrics.
        """
        logger.info(f"Running {self.name} pipeline for query: {query[:50]}...")
        start: float = time.perf_counter()
        
        try:
            context: str = self.retrieve_context(query)
            
            if context:
                system_msg = ("You are a precise medical assistant. Answer ONLY using the provided context. "
                             "If the context does not contain the answer, explicitly state that you do not know.")
                user_msg = f"Context:\n{context}\n\nQuestion: {query}"
            else:
                system_msg = "You are a concise medical assistant. Answer based on your internal knowledge."
                user_msg = query

            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.1,
                max_tokens=256
            )
            
            usage = response.usage
            tokens_p: int = usage.prompt_tokens
            tokens_c: int = usage.completion_tokens
            total: int = tokens_p + tokens_c
            latency: float = (time.perf_counter() - start) * 1000
            cost_usd: float = round((total / 1000) * COST_PER_1K, 6)
            
            logger.info(f"Tokens: {total} | Cost: ${cost_usd} | Latency: {latency:.0f}ms")
            
            return {
                "pipeline_name":     self.name,
                "answer":            response.choices[0].message.content.strip(),
                "tokens_prompt":     tokens_p,
                "tokens_completion": tokens_c,
                "total_tokens":      total,
                "latency_ms":        round(latency, 2),
                "cost_usd":          cost_usd
            }
            
        except Exception as e:
            logger.error(f"Error in {self.name} execution: {e}")
            return {
                "pipeline_name":     self.name,
                "answer":            "Error generating RAG response.",
                "tokens_prompt":     0,
                "tokens_completion": 0,
                "total_tokens":      0,
                "latency_ms":        0.0,
                "cost_usd":          0.0
            }
