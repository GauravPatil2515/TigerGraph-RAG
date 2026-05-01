"""Pipeline A: Raw LLM (LLM-Only) for GraphRAG Hackathon.

This pipeline makes direct API calls to Groq without any retrieval augmentation.
Serves as the baseline for comparison.
"""

import time
import logging
from typing import Dict, Any
from groq import Groq
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROQ_API_KEY, GROQ_MODEL, COST_PER_1K_TOKENS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RawLLMPipeline:
    """Raw LLM pipeline - direct Groq API calls without retrieval.
    
    This pipeline serves as the baseline for comparison with RAG pipelines.
    It sends queries directly to the LLM without any context retrieval.
    """
    
    def __init__(self):
        """Initialize the Raw LLM pipeline with Groq client."""
        self.client = Groq(api_key=GROQ_API_KEY)
        self.name = "llm_only"
    
    def __repr__(self) -> str:
        """Return string representation of the pipeline."""
        return f"RawLLMPipeline(name='{self.name}')"
    
    def run(self, query: str) -> Dict[str, Any]:
        """Run the raw LLM pipeline.
        
        Args:
            query: The medical question to answer
            
        Returns:
            Dict with pipeline_name, answer, tokens_prompt, tokens_completion,
            total_tokens, latency_ms, cost_usd
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful medical research assistant. Answer concisely and accurately based on your knowledge."
            },
            {"role": "user", "content": query}
        ]
        
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=512
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
            "cost_usd": round(cost_usd, 6)
        }


if __name__ == "__main__":
    logger.info("Testing Raw LLM Pipeline...")
    pipeline = RawLLMPipeline()
    logger.info(f"Pipeline: {pipeline}")
    
    test_query = "What are the symptoms of Type 2 Diabetes?"
    logger.info(f"\nTest query: {test_query}")
    
    result = pipeline.run(test_query)
    logger.info(f"\nResult:")
    for key, value in result.items():
        logger.info(f"  {key}: {value}")

