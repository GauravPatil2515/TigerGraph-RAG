"""Pipeline C: GraphRAG (TigerGraph REST API + Groq) for GraphRAG Hackathon.

This pipeline uses TigerGraph GraphRAG REST API for graph-based retrieval,
then augments the LLM prompt with graph context.
"""

import time
import logging
from typing import Dict, Any
from groq import Groq
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROQ_API_KEY, GROQ_MODEL, COST_PER_1K_TOKENS
from ingest import tigergraph_ingest as tg_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_compressed_context(graph_response: dict, max_chars: int = 800) -> str:
    """Extract only the essential graph entities and key facts — no raw passage dumps.

    GraphRAG's advantage is that the graph already distilled relationships.
    We take the shortest path to the answer: entities > key_facts > raw context.

    Args:
        graph_response: Dict returned from TigerGraph query_graphrag()
        max_chars: Hard cap on output length (default 800 chars ~ 200 tokens)

    Returns:
        Compressed context string
    """
    parts = []

    # 1. Named entities from graph traversal
    entities = graph_response.get("entities", [])
    if entities:
        entity_strs = []
        for e in entities[:5]:
            name = e.get("name") or e.get("id", "")
            desc = e.get("description") or e.get("summary", "")
            if name:
                entity_strs.append(f"{name}: {desc[:80]}" if desc else name)
        if entity_strs:
            parts.append("Entities: " + "; ".join(entity_strs))

    # 2. Relationships from graph traversal
    relationships = graph_response.get("relationships", [])
    if relationships:
        rel_strs = []
        for r in relationships[:4]:
            src = r.get("source") or r.get("from", "")
            tgt = r.get("target") or r.get("to", "")
            rel = r.get("relation") or r.get("type", "related")
            if src and tgt:
                rel_strs.append(f"{src} -{rel}-> {tgt}")
        if rel_strs:
            parts.append("Relations: " + "; ".join(rel_strs))

    # 3. Key facts / snippets from graph response
    key_facts = graph_response.get("key_facts") or graph_response.get("facts", [])
    if key_facts:
        facts_str = "; ".join(str(f)[:120] for f in key_facts[:3])
        parts.append(f"Facts: {facts_str}")

    # 4. Fall back to raw context only if nothing else available
    if not parts:
        raw = graph_response.get("context", "")
        if raw:
            parts.append(raw)

    compressed = "\n".join(parts)
    return compressed[:max_chars]


class GraphRAGPipeline:
    """GraphRAG pipeline using TigerGraph REST API for retrieval + Groq for generation.
    
    This pipeline retrieves relevant context from TigerGraph knowledge graph,
    then uses the graph context to augment the LLM prompt.
    """
    
    def __init__(self):
        """Initialize the GraphRAG pipeline with Groq client and TigerGraph headers."""
        self.client = Groq(api_key=GROQ_API_KEY)
        self.name = "graphrag"
    
    def __repr__(self) -> str:
        """Return string representation of the pipeline."""
        return f"GraphRAGPipeline(name='{self.name}')"
    
    def run(self, query: str) -> Dict[str, Any]:
        """Run the GraphRAG pipeline.
        
        Args:
            query: The medical question to answer
            
        Returns:
            Dict with pipeline_name, answer, tokens_prompt, tokens_completion,
            total_tokens, latency_ms, cost_usd, graph_context_tokens
        """
        start_total = time.perf_counter()
        
        # Call TigerGraph GraphRAG to get graph context
        try:
            graph_response = tg_client.query_graphrag(query, top_k=5, num_hops=2)
        except Exception as e:
            logger.error(f"TigerGraph query error: {e}")
            graph_response = {"error": str(e), "answer": None, "context": ""}
        
        # Check if GraphRAG returns a direct answer
        if graph_response.get("answer"):
            # GraphRAG REST returns a direct answer - use it, skip Groq call
            answer = graph_response["answer"]
            tokens_prompt = 0
            tokens_completion = 0
            total_tokens = 0
            cost_usd = 0.0
            graph_context_tokens = len(answer.split()) * 1.3  # Approximate
            latency_ms = (time.perf_counter() - start_total) * 1000
            
            return {
                "pipeline_name": self.name,
                "answer": answer,
                "tokens_prompt": tokens_prompt,
                "tokens_completion": tokens_completion,
                "total_tokens": total_tokens,
                "latency_ms": round(latency_ms, 2),
                "cost_usd": round(cost_usd, 6),
                "graph_context_tokens": round(graph_context_tokens, 0)
            }
        
        # GraphRAG returns context - pass to Groq
        # Use compressed extraction — graph already distilled key facts
        context = extract_compressed_context(graph_response, max_chars=800)
        if not context:
            context = "No relevant graph context found."
        
        # Calculate graph context tokens (compressed)
        graph_context_tokens = len(context.split())
        logger.info(f"GraphRAG compressed context: {len(context)} chars / ~{graph_context_tokens} words")
        
        # Build a tight prompt — graph context is pre-distilled, no need for verbose instructions
        messages = [
            {
                "role": "system",
                "content": "You are a concise medical assistant. Use the graph context below to answer factually in 1-3 sentences."
            },
            {"role": "user", "content": f"Graph context:\n{context}\n\nQuestion: {query}\nAnswer:"}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=250  # Tight — GraphRAG should be the most token-efficient
            )
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise
        
        latency_ms = (time.perf_counter() - start_total) * 1000
        
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
            "graph_context_tokens": graph_context_tokens
        }


if __name__ == "__main__":
    logger.info("Testing GraphRAG Pipeline...")
    pipeline = GraphRAGPipeline()
    logger.info(f"Pipeline: {pipeline}")
    
    # Check TigerGraph health first
    if tg_client.check_health():
        logger.info("✓ TigerGraph health check passed")
    else:
        logger.warning("✗ TigerGraph health check failed - pipeline may not work")
    
    test_query = "What are the symptoms of Type 2 Diabetes?"
    logger.info(f"\nTest query: {test_query}")
    
    try:
        result = pipeline.run(test_query)
        logger.info(f"\nResult:")
        for key, value in result.items():
            logger.info(f"  {key}: {value}")
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")

