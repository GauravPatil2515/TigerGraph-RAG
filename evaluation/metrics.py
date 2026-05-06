"""
Module: metrics.py
Description: Complementary evaluation metrics (ROUGE-L) for medical text 
             comparison. Provides overlap-based scoring to supplement 
             the semantic BERTScore evaluation.

Author: Gaurav Patil
Project: GraphRAG Inference Hackathon — TigerGraph 2026
"""

from rouge_score import rouge_scorer as rs
from bert_score import score as bert_score_fn
from typing import Dict

# Initialize the ROUGE scorer with focus on longest common subsequence (ROUGE-L)
_rouge = rs.RougeScorer(['rougeL'], use_stemmer=True)

def evaluate(prediction: str, reference: str) -> Dict[str, float]:
    """
    Evaluate prediction against reference using ROUGE and BERTScore.
    
    Args:
        prediction: The generated text.
        reference: The ground truth text.
        
    Returns:
        Dict[str, float]: Calculated metrics (rouge_l, bert_f1).
    """
    # Calculate overlap-based ROUGE score
    rouge = _rouge.score(reference, prediction)
    
    # Calculate semantic-based BERTScore
    # Note: Using default model for speed in this quick utility
    _, _, F1 = bert_score_fn([prediction], [reference], lang="en", verbose=False)
    
    return {
        "rouge_l":  round(rouge["rougeL"].fmeasure, 4),
        "bert_f1":  round(F1.mean().item(), 4)
    }
