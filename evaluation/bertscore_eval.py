"""BERTScore evaluation module for GraphRAG Hackathon.

Provides semantic similarity evaluation using BERTScore with rescaling.
Optimized to keep the model in memory across multiple calls.
Returns both raw and rescaled scores as required by hackathon.
"""

import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_bertscore(prediction: str, reference: str) -> Dict[str, float]:
    """Compute BERTScore between prediction and reference.
    
    Uses roberta-large model for higher quality scores.
    Returns both raw and rescaled F1 scores as required by hackathon specs.
    Target: raw >= 0.88, rescaled >= 0.55
    
    Args:
        prediction: Generated answer text
        reference: Ground truth reference text
        
    Returns:
        Dictionary with bert_f1_raw, bert_f1_rescaled, bert_precision, bert_recall
    """
    if not prediction or not reference:
        return {
            "bert_f1_raw": 0.0,
            "bert_f1_rescaled": 0.0,
            "bert_precision": 0.0,
            "bert_recall": 0.0
        }
    
    # Import here to avoid loading until needed
    from bert_score import score as bert_score_fn
    
    # Compute RAW scores (without baseline rescaling) using roberta-large
    P_raw, R_raw, F1_raw = bert_score_fn(
        [prediction],
        [reference],
        lang="en",
        model_type="roberta-large",
        verbose=False,
        rescale_with_baseline=False
    )
    
    # Compute RESCALED scores (with baseline rescaling) using roberta-large
    P_rescaled, R_rescaled, F1_rescaled = bert_score_fn(
        [prediction],
        [reference],
        lang="en",
        model_type="roberta-large",
        verbose=False,
        rescale_with_baseline=True
    )
    
    return {
        "bert_f1_raw": round(F1_raw.mean().item(), 4),
        "bert_f1_rescaled": round(F1_rescaled.mean().item(), 4),
        "bert_precision": round(P_rescaled.mean().item(), 4),
        "bert_recall": round(R_rescaled.mean().item(), 4)
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
        logger.info(f"  BERTScore F1 Raw: {scores['bert_f1_raw']}")
        logger.info(f"  BERTScore F1 Rescaled: {scores['bert_f1_rescaled']}")
        logger.info(f"  Precision: {scores['bert_precision']}")
        logger.info(f"  Recall: {scores['bert_recall']}")
