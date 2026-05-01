"""Benchmark runner for GraphRAG Hackathon.

Runs all pipelines on benchmark queries and logs results with evaluation metrics.
"""

import json
import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.bertscore_eval import compute_bertscore
from evaluation.llm_judge import llm_judge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Benchmark runner for comparing pipeline performance.
    
    Runs multiple pipelines on a set of benchmark queries, computes evaluation
    metrics (BERTScore, LLM Judge), and logs results to JSONL and CSV.
    """
    
    def __init__(self, pipelines: list, results_dir: str = "./results"):
        """Initialize the benchmark runner.
        
        Args:
            pipelines: List of pipeline instances to benchmark
            results_dir: Directory to save results (default: "./results")
        """
        self.pipelines = pipelines
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(results_dir, f"benchmark_{timestamp}.jsonl")
        self.csv_path = self.log_path.replace(".jsonl", ".csv")
        
        self.records: List[Dict[str, Any]] = []
        
        logger.info(f"BenchmarkRunner initialized")
        logger.info(f"Results will be saved to: {self.log_path}")
    
    def __repr__(self) -> str:
        """Return string representation of the runner."""
        pipeline_names = [p.name for p in self.pipelines]
        return f"BenchmarkRunner(pipelines={pipeline_names}, results_dir='{self.results_dir}')"
    
    def run(self, queries: List[Dict], delay_seconds: float = 1.5):
        """Run all pipelines on all benchmark queries.
        
        Args:
            queries: List of query dicts with id, hop_level, query, reference
            delay_seconds: Delay between API calls to avoid rate limiting (default: 1.5)
        """
        total_queries = len(queries)
        total_pipelines = len(self.pipelines)
        total_runs = total_queries * total_pipelines
        
        logger.info(f"Starting benchmark: {total_queries} queries × {total_pipelines} pipelines = {total_runs} total runs")
        
        run_count = 0
        for query in queries:
            logger.info(f"\n[Query {query['id']}] Hop:{query['hop_level']} — {query['query'][:60]}...")
            
            for pipeline in self.pipelines:
                run_count += 1
                logger.info(f"  Running {pipeline.name} ({run_count}/{total_runs})...")
                
                try:
                    # Run pipeline
                    result = pipeline.run(query["query"])
                    
                    # Compute BERTScore
                    bert_scores = compute_bertscore(result["answer"], query["reference"])
                    
                    # LLM Judge evaluation
                    judge_result = llm_judge(
                        query["query"],
                        result["answer"],
                        query["reference"]
                    )
                    
                    # Build complete record
                    record = {
                        "query_id": query["id"],
                        "hop_level": query["hop_level"],
                        "query": query["query"],
                        "reference": query["reference"],
                        "pipeline_name": result["pipeline_name"],
                        "answer": result["answer"],
                        "tokens_prompt": result.get("tokens_prompt", 0),
                        "tokens_completion": result.get("tokens_completion", 0),
                        "total_tokens": result.get("total_tokens", 0),
                        "latency_ms": result.get("latency_ms", 0),
                        "cost_usd": result.get("cost_usd", 0),
                        "bert_f1": bert_scores["bert_f1"],
                        "bert_precision": bert_scores["bert_precision"],
                        "bert_recall": bert_scores["bert_recall"],
                        "llm_judge_passed": judge_result["passed"],
                        "llm_judge_raw": judge_result["raw_response"],
                        "timestamp": time.time()
                    }
                    
                    # Add pipeline-specific fields
                    if "retrieved_docs" in result:
                        record["retrieved_docs_count"] = len(result["retrieved_docs"])
                    if "graph_context_tokens" in result:
                        record["graph_context_tokens"] = result["graph_context_tokens"]
                    
                    self.records.append(record)
                    
                    # Write to JSONL
                    with open(self.log_path, "a") as f:
                        f.write(json.dumps(record) + "\n")
                    
                    # Print progress
                    logger.info(
                        f"    ✓ {result['pipeline_name']}: "
                        f"tokens={record['total_tokens']}, "
                        f"latency={record['latency_ms']:.0f}ms, "
                        f"bert_f1={record['bert_f1']:.3f}, "
                        f"judge={'PASS' if record['llm_judge_passed'] else 'FAIL'}"
                    )
                    
                except Exception as e:
                    logger.error(f"    ✗ ERROR in {pipeline.name}: {e}")
                    # Write error record
                    error_record = {
                        "query_id": query["id"],
                        "hop_level": query["hop_level"],
                        "query": query["query"],
                        "pipeline_name": pipeline.name,
                        "error": str(e),
                        "timestamp": time.time()
                    }
                    with open(self.log_path, "a") as f:
                        f.write(json.dumps(error_record) + "\n")
                
                # Delay between API calls to avoid rate limiting
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
        
        logger.info(f"\nBenchmark complete. Processed {run_count} runs.")
    
    def save_csv(self) -> pd.DataFrame:
        """Save records to CSV file.
        
        Returns:
            DataFrame with all benchmark results
        """
        if not self.records:
            logger.warning("No records to save!")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.records)
        df.to_csv(self.csv_path, index=False)
        logger.info(f"Results saved to: {self.csv_path}")
        return df
    
    def print_summary(self, df: pd.DataFrame = None):
        """Print summary statistics by pipeline.
        
        Args:
            df: DataFrame to summarize (if None, uses self.records)
        """
        if df is None:
            if not self.records:
                logger.warning("No records to summarize!")
                return
            df = pd.DataFrame(self.records)
        
        logger.info("\n" + "="*80)
        logger.info("BENCHMARK SUMMARY")
        logger.info("="*80)
        
        # Group by pipeline and compute means
        numeric_cols = [
            "tokens_prompt", "tokens_completion", "total_tokens",
            "latency_ms", "cost_usd", "bert_f1", "bert_precision",
            "bert_recall"
        ]
        
        # Only include columns that exist
        available_cols = [col for col in numeric_cols if col in df.columns]
        summary = df.groupby("pipeline_name")[available_cols].mean()
        
        # Add pass rate for LLM judge
        if "llm_judge_passed" in df.columns:
            pass_rates = df.groupby("pipeline_name")["llm_judge_passed"].mean()
            summary["llm_judge_pass_rate"] = pass_rates
        
        logger.info("\nMean metrics by pipeline:")
        for pipeline_name in summary.index:
            logger.info(f"\n  {pipeline_name}:")
            for col in summary.columns:
                val = summary.loc[pipeline_name, col]
                if col in ["cost_usd"]:
                    logger.info(f"    {col}: ${val:.6f}")
                elif col in ["latency_ms"]:
                    logger.info(f"    {col}: {val:.1f} ms")
                elif col in ["tokens_prompt", "tokens_completion", "total_tokens"]:
                    logger.info(f"    {col}: {val:.0f}")
                else:
                    logger.info(f"    {col}: {val:.4f}")
        
        logger.info("\n" + "="*80)


if __name__ == "__main__":
    # Test the runner
    from pipelines.pipeline_a_raw_llm import RawLLMPipeline
    from benchmark.queries import BENCHMARK_QUERIES
    
    logger.info("Testing BenchmarkRunner...")
    
    # Create a test pipeline
    pipeline = RawLLMPipeline()
    runner = BenchmarkRunner([pipeline], results_dir="./results")
    
    # Run on just 3 queries
    test_queries = BENCHMARK_QUERIES[:3]
    runner.run(test_queries, delay_seconds=0.5)
    
    # Save and summarize
    df = runner.save_csv()
    runner.print_summary(df)

