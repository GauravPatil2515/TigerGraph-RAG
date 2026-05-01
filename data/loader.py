"""PubMedQA dataset loader for GraphRAG Hackathon.

This module handles loading and preprocessing of the PubMedQA dataset from HuggingFace.
"""

import logging
from typing import List, Dict, Any
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pubmedqa(limit: int = 1000) -> List[Dict[str, Any]]:
    """Load PubMedQA dataset from HuggingFace.
    
    Args:
        limit: Maximum number of records to load (default: 1000)
        
    Returns:
        List of dicts with keys: question, answer, context, token_count
    """
    logger.info(f"Loading PubMedQA dataset (pqa_labeled split, {limit} records)...")
    
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    records = []
    
    for i, row in enumerate(ds):
        if i >= limit:
            break
        
        # Extract question
        question = row.get("question", "").strip()
        
        # Extract long answer (reference answer)
        answer = row.get("long_answer", "").strip()
        
        # Extract context - join context.contexts list into one string
        context_list = row.get("context", {}).get("contexts", [])
        context = " ".join(context_list).strip()
        
        # Calculate token count estimate: len(text.split()) * 1.3
        total_text = f"{question} {answer} {context}"
        token_count = len(total_text.split()) * 1.3
        
        if question and answer:  # Only include records with both question and answer
            records.append({
                "question": question,
                "answer": answer,
                "context": context,
                "token_count": round(token_count, 0)
            })
    
    # Calculate total estimated tokens
    total_tokens = sum(r["token_count"] for r in records)
    logger.info(f"Loaded {len(records)} QA pairs from PubMedQA")
    logger.info(f"Total estimated tokens: {total_tokens:,.0f} (~{total_tokens/1e6:.2f}M tokens)")
    
    return records


if __name__ == "__main__":
    data = load_pubmedqa()
    logger.info(f"\nSample record:")
    if data:
        sample = data[0]
        logger.info(f"  Question: {sample['question'][:100]}...")
        logger.info(f"  Answer: {sample['answer'][:100]}...")
        logger.info(f"  Context: {sample['context'][:100]}...")
        logger.info(f"  Token count: {sample['token_count']}")
