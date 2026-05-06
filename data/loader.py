"""
Module: loader.py
Description: PubMedQA dataset loader for GraphRAG Hackathon. 
             Handles high-speed ingestion and preprocessing of medical QA 
             data from HuggingFace.

Author: Gaurav Patil
Project: GraphRAG Inference Hackathon — TigerGraph 2026

Key Features:
    - Automated fetching via 'datasets' library
    - Context merging and sanitization
    - Heuristic token count estimation for ingestion planning
"""

import logging
from typing import List, Dict, Any

# Configure structured logging
logger = logging.getLogger(__name__)

def load_pubmedqa(limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Load and preprocess the PubMedQA dataset from HuggingFace.
    
    Fetches the 'pqa_labeled' split which contains high-quality, 
    human-labeled medical questions, contexts, and answers.

    Args:
        limit: Maximum number of records to load (default: 1000).
        
    Returns:
        List[Dict[str, Any]]: List of records, each containing:
            - question: The medical inquiry
            - answer: The ground truth long answer
            - context: Concatenated supporting document text
            - token_count: Estimated token usage (words * 1.3)
            
    Example:
        records = load_pubmedqa(limit=100)
        print(records[0]['question'])
    """
    logger.info(f"Fetching PubMedQA dataset (pqa_labeled split, limit={limit})...")
    
    try:
        from datasets import load_dataset
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return []

    records: List[Dict[str, Any]] = []
    
    for i, row in enumerate(ds):
        if i >= limit:
            break
        
        # Extract and sanitize primary fields
        question: str = row.get("question", "").strip()
        answer: str   = row.get("long_answer", "").strip()
        
        # Merge context snippets into a single coherent block for RAG
        context_list: List[str] = row.get("context", {}).get("contexts", [])
        context: str = " ".join(context_list).strip()
        
        # Calculate heuristic token count estimate (words * 1.3 coefficient)
        # This is used to estimate ingestion costs and context window usage.
        total_text: str = f"{question} {answer} {context}"
        token_count: float = len(total_text.split()) * 1.3
        
        # Filter for quality: only include records with complete QA pairs
        if question and answer:
            records.append({
                "question": question,
                "answer": answer,
                "context": context,
                "token_count": round(token_count, 0)
            })
    
    total_tokens: float = sum(r["token_count"] for r in records)
    logger.info(f"✅ Successfully loaded {len(records)} records")
    logger.info(f"📊 Total estimated tokens: {total_tokens:,.0f} (~{total_tokens/1e6:.2f}M tokens)")
    
    return records

if __name__ == "__main__":
    # Test execution for standalone verification
    logging.basicConfig(level=logging.INFO)
    data = load_pubmedqa(5)
    if data:
        logger.info(f"Sample Question: {data[0]['question'][:80]}...")
