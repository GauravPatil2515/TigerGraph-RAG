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
    
    def __init__(self, pipelines: list, results_dir: str = "./results", mode: str = "quick"):
        """Initialize the benchmark runner.
        
        Args:
            pipelines: List of pipeline instances to benchmark
            results_dir: Directory to save results (default: "./results")
            mode: "quick" (30 queries) or "full" (200 PubMedQA records)
        """
        self.pipelines = pipelines
        self.results_dir = results_dir
        self.mode = mode
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if mode == "full":
            self.log_path = os.path.join(results_dir, f"full_benchmark_{timestamp}.jsonl")
        else:
            self.log_path = os.path.join(results_dir, f"benchmark_{timestamp}.jsonl")
        self.csv_path = self.log_path.replace(".jsonl", ".csv")
        
        self.records: List[Dict[str, Any]] = []
        
        logger.info(f"BenchmarkRunner initialized (mode: {mode})")
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
                    
                    # Build complete record with all required columns
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
                        "bert_f1_raw": bert_scores["bert_f1_raw"],
                        "bert_f1_rescaled": bert_scores["bert_f1_rescaled"],
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
                        f"bert_f1_raw={record['bert_f1_raw']:.3f}, "
                        f"bert_f1_rescaled={record['bert_f1_rescaled']:.3f}, "
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
            "latency_ms", "cost_usd", "bert_f1_raw", "bert_f1_rescaled",
            "bert_precision", "bert_recall"
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
    
    def run_full_benchmark(self, records: List[Dict], pipelines: List, delay_seconds: float = 0.5) -> Dict:
        """Run full benchmark on PubMedQA records (200 queries).
        
        Optimized version that runs only LLM-Only and GraphRAG pipelines
        with BERTScore evaluation only (skips LLM Judge for speed).
        
        Args:
            records: List of PubMedQA records with question and answer
            pipelines: List of pipeline instances (only uses llm_only and graphrag)
            delay_seconds: Delay between API calls (default: 0.5)
            
        Returns:
            Dict with summary statistics per pipeline
        """
        # Filter to first 200 records
        records = records[:200]
        
        # Filter pipelines to only llm_only and graphrag
        filtered_pipelines = [p for p in pipelines if p.name in ["llm_only", "graphrag"]]
        if not filtered_pipelines:
            logger.error("No valid pipelines for full benchmark (need llm_only and/or graphrag)")
            return {}
        
        logger.info(f"\n{'='*80}")
        logger.info("FULL BENCHMARK MODE (200 PubMedQA records)")
        logger.info(f"{'='*80}")
        logger.info(f"Pipelines: {[p.name for p in filtered_pipelines]}")
        logger.info(f"Records: {len(records)}")
        logger.info(f"Delay: {delay_seconds}s between calls")
        logger.info(f"Note: BERTScore only (LLM Judge skipped for speed)")
        logger.info(f"{'='*80}\n")
        
        run_count = 0
        total_runs = len(records) * len(filtered_pipelines)
        
        for i, record in enumerate(records):
            # Print progress every 10 records
            if i % 10 == 0:
                logger.info(f"Progress: {i}/200 records processed ({i/2:.0f}%)")
            
            question = record.get("question", "")
            reference = record.get("answer", "")
            
            for pipeline in filtered_pipelines:
                run_count += 1
                
                try:
                    # Run pipeline
                    result = pipeline.run(question)
                    
                    # Compute BERTScore only (skip LLM Judge for speed)
                    bert_scores = compute_bertscore(result["answer"], reference)
                    
                    # Build record
                    rec = {
                        "record_idx": i,
                        "query": question,
                        "reference": reference[:200],  # Truncate for storage
                        "pipeline_name": result["pipeline_name"],
                        "answer": result["answer"][:500],  # Truncate for storage
                        "tokens_prompt": result.get("tokens_prompt", 0),
                        "tokens_completion": result.get("tokens_completion", 0),
                        "total_tokens": result.get("total_tokens", 0),
                        "latency_ms": result.get("latency_ms", 0),
                        "cost_usd": result.get("cost_usd", 0),
                        "bert_f1_raw": bert_scores["bert_f1_raw"],
                        "bert_f1_rescaled": bert_scores["bert_f1_rescaled"],
                        "bert_precision": bert_scores["bert_precision"],
                        "bert_recall": bert_scores["bert_recall"],
                        "timestamp": time.time()
                    }
                    
                    self.records.append(rec)
                    
                    # Write to JSONL
                    with open(self.log_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    
                except Exception as e:
                    logger.error(f"    ✗ ERROR in {pipeline.name} on record {i}: {e}")
                
                # Delay between API calls
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
        
        logger.info(f"\nFull benchmark complete. Processed {run_count}/{total_runs} runs.")
        
        # Return summary
        if self.records:
            df = pd.DataFrame(self.records)
            summary = df.groupby("pipeline_name").agg({
                "total_tokens": "mean",
                "latency_ms": "mean",
                "cost_usd": "mean",
                "bert_f1_rescaled": "mean"
            }).to_dict()
            return summary
        return {}


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

