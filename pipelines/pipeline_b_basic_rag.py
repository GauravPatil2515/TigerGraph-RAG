"""
Module: pipeline_b_basic_rag.py
Description: Pipeline B — standard vector RAG using ChromaDB for similarity search.
             Represents the current industry standard baseline.

Author: Gaurav Patil
Project: GraphRAG Inference Hackathon — TigerGraph 2026
GitHub: https://github.com/GauravPatil2515/TigerGraph-RAG

Key Features:
    - ChromaDB vector retrieval using Sentence Transformers
    - Relevance-based filtering (distance < 1.2)
    - FIXED: retrieves stored answer from metadata (not raw document chunk)
      so token comparison vs GraphRAG is fair and defensible
"""

import time
import logging
import os
import sys
from typing import Dict, Any
from groq import Groq

sys.path.insert(0, os.getcwd())
from config import GROQ_API_KEY, GROQ_MODEL, COST_PER_1K
from ingest.chroma_ingest import get_collection

logger = logging.getLogger(__name__)


class BasicRAGPipeline:
    """
    BasicRAGPipeline: Pipeline B of the 3-pipeline benchmark system.

    Implements standard vector-based RAG using ChromaDB. Retrieves the stored
    ground-truth answer snippets from document metadata (not raw doc chunks)
    to provide a fair token-efficiency comparison against GraphRAG.

    Attributes:
        client (Groq): Initialized Groq LLM client
        name (str): Pipeline identifier for CSV logging
        collection: ChromaDB collection handle for 'pubmedqa'
    """

    def __init__(self) -> None:
        self.client = Groq(api_key=GROQ_API_KEY)
        self.name: str = "basic_rag"
        self.collection = get_collection()
        logger.info(f"✅ {self.name} pipeline initialized")

    def retrieve_context(self, query: str) -> str:
        """
        Retrieve relevant context from ChromaDB with relevance filtering.

        FIXED: Now retrieves the stored 'answer' field from metadata instead
        of the raw document chunk (question + context preview). This ensures
        the Basic RAG pipeline feeds the LLM actual answer-quality context,
        making the token-efficiency comparison with GraphRAG fair and accurate.

        Args:
            query: User's medical question text.

        Returns:
            str: Concatenated answer snippets from top-k relevant documents.
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=3,
                include=["documents", "distances", "metadatas"]
            )

            if not results["documents"] or not results["documents"][0]:
                return ""

            docs      = results["documents"][0]
            dists     = results["distances"][0]
            metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)

            RELEVANCE_THRESHOLD: float = 1.2
            relevant = []

            for doc, dist, meta in zip(docs, dists, metadatas):
                if dist < RELEVANCE_THRESHOLD:
                    # Use stored answer from metadata for fair context comparison
                    # Falls back to raw document chunk if metadata answer unavailable
                    answer_text = meta.get("answer", "") or doc
                    if answer_text:
                        relevant.append(answer_text[:500])
                else:
                    logger.debug(f"Skipping irrelevant snippet: distance={dist:.4f}")

            return "\n\n---\n\n".join(relevant[:2]) if relevant else ""

        except Exception as e:
            logger.error(f"ChromaDB retrieval error in {self.name}: {e}")
            return ""

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the Basic RAG pipeline for a given query.

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
                system_msg = (
                    "You are a precise medical assistant. Answer ONLY using the provided context. "
                    "If the context does not contain the answer, state that you do not know."
                )
                user_msg = f"Context:\n{context}\n\nQuestion: {query}"
            else:
                system_msg = "You are a concise medical assistant. Answer based on your internal knowledge."
                user_msg   = query

            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg}
                ],
                temperature=0.1,
                max_tokens=256
            )

            usage       = response.usage
            tokens_p: int   = usage.prompt_tokens
            tokens_c: int   = usage.completion_tokens
            total: int      = tokens_p + tokens_c
            latency: float  = (time.perf_counter() - start) * 1000
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
                "tokens_prompt":     0, "tokens_completion": 0, "total_tokens": 0,
                "latency_ms":        0.0, "cost_usd": 0.0
            }
