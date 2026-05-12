"""
Module: pipeline_a_raw_llm.py
Description: Baseline pipeline — no retrieval, direct LLM inference. 
             Serves as the worst-case cost/accuracy baseline for comparison 
             against RAG-based approaches.

Author: Gaurav Patil
Project: GraphRAG Inference Hackathon — TigerGraph 2026
GitHub: https://github.com/GauravPatil2515/TigerGraph-RAG

Key Features:
    - Zero-shot inference using Groq Llama-3
    - Establishing baseline metrics for token usage and latency
    - No external knowledge retrieval
    - temperature=0.1 (consistent with all other pipelines for fair benchmarking)
"""

import time
import logging
import os
import sys
from typing import Dict, Any
from groq import Groq

sys.path.insert(0, os.getcwd())
from config import GROQ_API_KEY, GROQ_MODEL, COST_PER_1K

logger = logging.getLogger(__name__)

class RawLLMPipeline:
    """
    RawLLMPipeline: Pipeline A of the 3-pipeline benchmark system.

    Implements a direct LLM call without any retrieval augmentation.
    Used to measure the 'pure' intelligence and cost baseline of the model.

    Attributes:
        client (Groq): Initialized Groq LLM client
        name (str): Pipeline identifier for CSV logging
    """
    
    def __init__(self) -> None:
        """Initialize the Raw LLM pipeline with Groq client."""
        self.client = Groq(api_key=GROQ_API_KEY)
        self.name: str = "llm_only"
        logger.info(f"✅ {self.name} pipeline initialized")

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the raw LLM pipeline for a given query.

        Args:
            query: User's medical question text.

        Returns:
            dict: Standard pipeline result dictionary.
        """
        logger.info(f"Running {self.name} pipeline for query: {query[:50]}...")
        start: float = time.perf_counter()
        
        try:
            # FIX: temperature=0.1 (was 0.2) — unified across all pipelines for fair comparison
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a concise medical assistant. Answer only based on your internal knowledge."},
                    {"role": "user", "content": query}
                ],
                temperature=0.1,
                max_tokens=512
            )
            
            usage = response.usage
            tokens_p: int  = usage.prompt_tokens
            tokens_c: int  = usage.completion_tokens
            total: int     = tokens_p + tokens_c
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
            logger.error(f"Groq API error in {self.name}: {e}")
            return {
                "pipeline_name":     self.name,
                "answer":            f"Error generating response: {str(e)}",
                "tokens_prompt":     0, "tokens_completion": 0, "total_tokens": 0,
                "latency_ms":        0.0, "cost_usd": 0.0
            }
