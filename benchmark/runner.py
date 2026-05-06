"""
Module: runner.py
Description: Orchestrates the execution of multiple GraphRAG pipelines across
             the benchmark query set. Computes metrics and exports results.

Author: Gaurav Patil
Project: GraphRAG Inference Hackathon — TigerGraph 2026

CSV SCHEMA (results/benchmark_*.csv):
    query_id         - unique query identifier
    hop_level        - query complexity: 1 (simple), 2 (relationship), 3 (multi-hop)
    query            - the medical question text
    reference        - ground truth answer for evaluation
    pipeline_name    - "llm_only" | "basic_rag" | "graphrag"
    answer           - LLM-generated response
    tokens_prompt    - input tokens used
    tokens_completion - output tokens generated
    total_tokens     - tokens_prompt + tokens_completion
    latency_ms       - end-to-end pipeline latency in milliseconds
    cost_usd         - estimated API cost in USD
    bert_f1_raw      - BERTScore F1 without baseline rescaling
    bert_f1_rescaled - BERTScore F1 with baseline rescaling (human-calibrated)
    bert_precision   - BERTScore precision
    bert_recall      - BERTScore recall
    llm_judge_passed - True/False verdict from LLM evaluator
    llm_judge_raw    - raw LLM judge response text
    efficiency_score - composite metric (accuracy * 0.6 + token_efficiency * 0.4)
    timestamp        - ISO 8601 run timestamp
"""

import json
import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import sys

# Ensure project root is in path for config import
sys.path.insert(0, os.getcwd())
from evaluation.bertscore_eval import compute_bertscore
from evaluation.llm_judge import llm_judge

# Configure structured logging
logger = logging.getLogger(__name__)

def compute_efficiency_score(row: pd.Series) -> float:
    """
    Compute composite efficiency score combining token savings + accuracy.
    
    Formula: (accuracy_score * 0.6) + (token_efficiency * 0.4)
    Where:
        accuracy_score  = bert_f1_raw (0 to 1)
        token_efficiency = 1 - (total_tokens / baseline_tokens)
    
    Args:
        row: DataFrame row with total_tokens and bert_f1_raw columns.
        
    Returns:
        float: Efficiency score between 0 and 1. Higher is better.
               GraphRAG target: > 0.75
    """
    accuracy: float    = row.get("bert_f1_raw", 0.0)
    tokens: float      = row.get("total_tokens", 1000.0)
    baseline: float    = 678.0   # Basic RAG average tokens (measured)
    
    # Calculate token efficiency component
    token_efficiency: float  = max(0.0, 1.0 - (tokens / baseline))
    
    # Composite score calculation
    return round((accuracy * 0.6) + (token_efficiency * 0.4), 4)

class BenchmarkRunner:
    """
    BenchmarkRunner: Core engine for evaluating RAG pipeline performance.
    
    Handles the execution loop for multiple pipelines, collects telemetry,
    and performs automated evaluation using BERTScore and LLM-as-a-Judge.
    
    Attributes:
        pipelines (list): Instances of pipelines to be tested.
        results_dir (str): Location to store CSV output.
        csv_path (str): Final path of the current benchmark session file.
        records (list): List of result dictionaries.
    """
    
    def __init__(self, pipelines: List[Any], results_dir: str = "./results") -> None:
        """
        Initialize the Benchmark Runner.

        Args:
            pipelines: List of pipeline objects (must implement .run() and have .name).
            results_dir: Directory to save performance results.
        """
        self.pipelines: List[Any] = pipelines
        self.results_dir: str = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path: str = os.path.join(results_dir, f"benchmark_{timestamp}.csv")
        self.records: List[Dict[str, Any]] = []
        logger.info(f"🚀 BenchmarkRunner initialized with {len(self.pipelines)} pipelines")

    def run(self, queries: List[Dict[str, Any]], delay: float = 1.0) -> None:
        """
        Run all pipelines on a set of benchmark queries.

        Args:
            queries: List of query dictionaries with question and reference.
            delay: Pause between API calls to avoid rate limiting.
        """
        total: int = len(queries) * len(self.pipelines)
        count: int = 0
        
        for i, q in enumerate(queries):
            query_id: str = str(q.get("id", "0"))
            hop_level: int = q.get("hop_level", 1)
            text: str = q.get("query", "")
            ref: str = q.get("reference", "")
            
            logger.info(f"Query [{i+1}/{len(queries)}]: {text[:50]}...")
            
            for pipe in self.pipelines:
                count += 1
                logger.info(f"  Running {pipe.name}...")
                
                try:
                    res: Dict[str, Any] = pipe.run(text)
                    
                    # Evaluate results using semantic and factual metrics
                    bs: Dict[str, float] = compute_bertscore(res.get("answer", ""), ref)
                    judge: Dict[str, Any] = llm_judge(text, res.get("answer", ""), ref)
                    
                    record = {
                        "query_id":          query_id,
                        "hop_level":         hop_level,
                        "query":             text,
                        "reference":         ref,
                        "pipeline_name":     pipe.name,
                        "answer":            res.get("answer", ""),
                        "tokens_prompt":     res.get("tokens_prompt", 0),
                        "tokens_completion": res.get("tokens_completion", 0),
                        "total_tokens":      res.get("total_tokens", 0),
                        "latency_ms":        res.get("latency_ms", 0.0),
                        "cost_usd":          res.get("cost_usd", 0.0),
                        "bert_f1_raw":       bs.get("bert_f1_raw", 0.0),
                        "bert_f1_rescaled":  bs.get("bert_f1_rescaled", 0.0),
                        "bert_precision":    bs.get("bert_precision", 0.0),
                        "bert_recall":       bs.get("bert_recall", 0.0),
                        "llm_judge_passed":  judge.get("passed", False),
                        "llm_judge_raw":     judge.get("raw_response", ""),
                        "timestamp":         datetime.now().isoformat()
                    }
                    self.records.append(record)
                    
                except Exception as e:
                    logger.error(f"Runner Error on {pipe.name} for query {query_id}: {e}")
                    # Save failure record to maintain dataset balance
                    self.records.append({
                        "query_id": query_id, "hop_level": hop_level, "query": text, "reference": ref,
                        "pipeline_name": pipe.name, "answer": f"ERROR: {e}",
                        "tokens_prompt": 0, "tokens_completion": 0, "total_tokens": 0,
                        "latency_ms": 0.0, "cost_usd": 0.0,
                        "bert_f1_raw": 0.0, "bert_f1_rescaled": 0.0, "bert_precision": 0.0, "bert_recall": 0.0,
                        "llm_judge_passed": False, "llm_judge_raw": str(e), "timestamp": datetime.now().isoformat()
                    })
                
                time.sleep(delay)

    def save(self) -> Optional[str]:
        """
        Finalize and save benchmark results to CSV.
        
        Applies efficiency score calculations and ensures schema consistency.

        Returns:
            str: Absolute path to the generated CSV file.
        """
        if not self.records:
            logger.warning("No records collected to save.")
            return None
            
        df: pd.DataFrame = pd.DataFrame(self.records)
        
        # IMPROVEMENT 3: Token Efficiency Score calculation
        df["efficiency_score"] = df.apply(compute_efficiency_score, axis=1)
        
        # Ensure exact column order and existence as per production schema
        cols: List[str] = [
            "query_id", "hop_level", "query", "reference", "pipeline_name", "answer",
            "tokens_prompt", "tokens_completion", "total_tokens", "latency_ms", "cost_usd",
            "bert_f1_raw", "bert_f1_rescaled", "bert_precision", "bert_recall",
            "llm_judge_passed", "llm_judge_raw", "efficiency_score", "timestamp"
        ]
        
        # Backfill any missing columns (e.g. from error records) with default zero values
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
                
        df = df[cols]
        df.to_csv(self.csv_path, index=False)
        logger.info(f"✅ Benchmark results saved to {self.csv_path}")
        return self.csv_path
