"""BERTScore evaluation module for GraphRAG Hackathon.

Provides semantic similarity evaluation using BERTScore with rescaling.
Optimized to keep the model in memory across multiple calls.
"""

import logging
from typing import Dict
from bert_score import BERTScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module-level singleton for the scorer
_scorer = None

def get_scorer():
    global _scorer
    if _scorer is None:
        logger.info("Initializing BERTScorer with bert-base-uncased...")
        _scorer = BERTScorer(
            model_type="bert-base-uncased",
            lang="en",
            rescale_with_baseline=True
        )
    return _scorer

def compute_bertscore(prediction: str, reference: str) -> Dict[str, float]:
    """Compute BERTScore F1 between prediction and reference.
    
    Args:
        prediction: Generated answer text
        reference: Reference answer text
        
    Returns:
        Dict with bert_f1, bert_precision, bert_recall (all rescaled)
    """
    try:
        scorer = get_scorer()
        # Compute BERTScore
        P, R, F1 = scorer.score([prediction], [reference])
        
        return {
            "bert_f1": round(F1.mean().item(), 4),
            "bert_precision": round(P.mean().item(), 4),
            "bert_recall": round(R.mean().item(), 4)
        }
    except Exception as e:
        logger.error(f"BERTScore computation error: {e}")
        return {
            "bert_f1": 0.0,
            "bert_precision": 0.0,
            "bert_recall": 0.0
        }

if __name__ == "__main__":
    logger.info("Testing BERTScore evaluation...")
    
    # Test cases
    test_cases = [
        {
            "prediction": "The symptoms of diabetes include increased thirst and frequent urination.",
            "reference": "Diabetes symptoms include increased thirst, frequent urination, fatigue, and blurred vision."
        }
    ]
    
    for i, test in enumerate(test_cases):
        logger.info(f"\nTest case {i+1}:")
        scores = compute_bertscore(test["prediction"], test["reference"])
        logger.info(f"  BERTScore F1: {scores['bert_f1']}")
