"""Pipeline B: Basic RAG (ChromaDB + Groq) for GraphRAG Hackathon.

This pipeline uses ChromaDB for vector similarity search, then augments
the LLM prompt with retrieved context.
"""

import time
import logging
from typing import Dict, Any, List
from groq import Groq
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROQ_API_KEY, GROQ_MODEL, COST_PER_1K_TOKENS
from ingest.chroma_ingest import get_collection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicRAGPipeline:
    """Basic RAG pipeline using ChromaDB for retrieval + Groq for generation.
    
    This pipeline retrieves relevant documents from ChromaDB vector store,
    then uses the retrieved context to augment the LLM prompt.
    """
    
    def __init__(self):
        """Initialize the Basic RAG pipeline with Groq client and ChromaDB collection."""
        self.client = Groq(api_key=GROQ_API_KEY)
        self.name = "basic_rag"
        self.collection = get_collection()
    
    def __repr__(self) -> str:
        """Return string representation of the pipeline."""
        return f"BasicRAGPipeline(name='{self.name}')"
    
    def run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Run the Basic RAG pipeline.
        
        Args:
            query: The medical question to answer
            top_k: Number of documents to retrieve (default: 5)
            
        Returns:
            Dict with pipeline_name, answer, tokens_prompt, tokens_completion,
            total_tokens, latency_ms, cost_usd, retrieved_docs
        """
        # Query ChromaDB for top_k most similar docs
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
        except Exception as e:
            logger.error(f"ChromaDB query error: {e}")
            raise
        
        # Build context from retrieved docs — use question + answer + context_preview
        retrieved_docs: List[str] = []
        if results["metadatas"] and results["metadatas"][0]:
            for metadata in results["metadatas"][0]:
                if not metadata:
                    continue
                parts = []
                if metadata.get("question"):
                    parts.append(f"Q: {metadata['question']}")
                if metadata.get("answer"):
                    parts.append(f"A: {metadata['answer'][:300]}")
                if metadata.get("context_preview"):
                    parts.append(f"Context: {metadata['context_preview'][:200]}")
                if parts:
                    retrieved_docs.append("\n".join(parts))
        
        if not retrieved_docs:
            logger.warning("ChromaDB returned no relevant documents for query!")
        
        # Cap context at ~2500 chars (~600 tokens) — enough info, token-efficient
        context = "\n\n---\n\n".join(retrieved_docs)[:2500]
        
        # Build prompt with context
        if context:
            user_msg = f"Use the following medical research context to answer the question concisely.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        else:
            user_msg = f"Answer the following medical question concisely based on your knowledge.\n\nQuestion: {query}\n\nAnswer:"
        
        messages = [
            {
                "role": "system",
                "content": "You are a concise medical research assistant. Answer questions using the provided context. Be direct and factual. Keep answers under 200 words."
            },
            {"role": "user", "content": user_msg}
        ]
        
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=300  # Keep completion tight
            )
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        usage = response.usage
        tokens_prompt = usage.prompt_tokens
        tokens_completion = usage.completion_tokens
        total_tokens = tokens_prompt + tokens_completion
        cost_usd = (total_tokens / 1000) * COST_PER_1K_TOKENS
        
        return {
            "pipeline_name": self.name,
            "answer": response.choices[0].message.content.strip(),
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "total_tokens": total_tokens,
            "latency_ms": round(latency_ms, 2),
            "cost_usd": round(cost_usd, 6),
            "retrieved_docs": retrieved_docs
        }


if __name__ == "__main__":
    logger.info("Testing Basic RAG Pipeline...")
    pipeline = BasicRAGPipeline()
    logger.info(f"Pipeline: {pipeline}")
    
    test_query = "What are the symptoms of Type 2 Diabetes?"
    logger.info(f"\nTest query: {test_query}")
    
    try:
        result = pipeline.run(test_query)
        logger.info(f"\nResult:")
        for key, value in result.items():
            if key == "retrieved_docs":
                logger.info(f"  {key}: {len(value)} documents")
                for i, doc in enumerate(value[:2]):
                    logger.info(f"    Doc {i+1}: {doc[:100]}...")
            else:
                logger.info(f"  {key}: {value}")
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
