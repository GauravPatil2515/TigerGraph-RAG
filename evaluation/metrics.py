# evaluation/metrics.py
from rouge_score import rouge_scorer as rs
from bert_score import score as bert_score_fn

_rouge = rs.RougeScorer(['rougeL'], use_stemmer=True)

def evaluate(prediction: str, reference: str) -> dict:
    rouge = _rouge.score(reference, prediction)
    _, _, F1 = bert_score_fn([prediction], [reference], lang="en", verbose=False)
    return {
        "rouge_l":  round(rouge["rougeL"].fmeasure, 4),
        "bert_f1":  round(F1.mean().item(), 4)
    }
