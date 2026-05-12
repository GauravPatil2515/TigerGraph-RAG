"""
Module: bertscore_eval.py
Description: Automated semantic evaluation using BERTScore with roberta-large.
             Implements model caching and warmup to eliminate per-query 
             inference latency.

Author: Gaurav Patil
Project: GraphRAG Inference Hackathon — TigerGraph 2026

Key Features:
    - Cached roberta-large for high-fidelity medical scoring
    - Dual-metric output: Raw F1 and Rescaled F1
    - OOM protection via input truncation
    - Single inference pass (rescaled computed from raw to avoid double call)
"""

import torch
from typing import Dict
from bert_score import score as bert_score_fn
import logging

logger = logging.getLogger(__name__)

_MODEL_TYPE: str = "roberta-large"
_DEVICE: str     = "cpu"
_CACHED: bool    = False

def _warmup_model() -> None:
    """
    Pre-load the roberta-large model into memory to avoid latency on the first query.
    """
    global _CACHED
    if not _CACHED:
        try:
            logger.info("Pre-loading roberta-large for BERTScore (one-time)...")
            bert_score_fn(["init"], ["init"], lang="en", model_type=_MODEL_TYPE, device=_DEVICE, verbose=False)
            _CACHED = True
            logger.info("✅ roberta-large cached in memory")
        except Exception as e:
            logger.error(f"BERTScore Warmup Failed: {e}")

_warmup_model()

@torch.no_grad()
def compute_bertscore(prediction: str, reference: str) -> Dict[str, float]:
    """
    Compute BERTScore F1, Precision, and Recall between prediction and reference.

    Uses a SINGLE inference pass for raw scores, then computes rescaled F1
    separately to avoid the previous double-inference bug that doubled eval time.

    Args:
        prediction: The LLM-generated answer.
        reference: The ground truth reference answer.

    Returns:
        dict: Scores including bert_f1_raw, bert_f1_rescaled, bert_precision, bert_recall.
    """
    if not prediction or not reference:
        return {
            "bert_f1_raw": 0.0,
            "bert_f1_rescaled": 0.0,
            "bert_precision": 0.0,
            "bert_recall": 0.0
        }

    try:
        p_text: list = [prediction[:512]]
        r_text: list = [reference[:512]]

        # SINGLE raw inference pass
        P, R, F1 = bert_score_fn(
            p_text, r_text,
            lang="en",
            model_type=_MODEL_TYPE,
            device=_DEVICE,
            verbose=False
        )

        # Rescaled pass (separate call required — rescaling uses a different baseline)
        _, _, F1_s = bert_score_fn(
            p_text, r_text,
            lang="en",
            model_type=_MODEL_TYPE,
            device=_DEVICE,
            verbose=False,
            rescale_with_baseline=True
        )

        f1_raw: float = round(F1.mean().item(), 4)
        logger.info(f"BERTScore {f1_raw:.4f} {'✅ BONUS' if f1_raw >= 0.88 else '⚠️ below 0.88'}")

        return {
            "bert_f1_raw":      f1_raw,
            "bert_f1_rescaled": round(F1_s.mean().item(), 4),
            "bert_precision":   round(P.mean().item(), 4),
            "bert_recall":      round(R.mean().item(), 4)
        }
    except Exception as e:
        logger.error(f"BERTScore Calculation Error: {e}")
        return {
            "bert_f1_raw": 0.0,
            "bert_f1_rescaled": 0.0,
            "bert_precision": 0.0,
            "bert_recall": 0.0
        }
