"""
Module: main.py
Description: Main entrypoint for GraphRAG Hackathon project. 
             Handles orchestration of data loading, ingestion, and 
             benchmark execution across multiple pipelines.

Author: Gaurav Patil
Project: GraphRAG Inference Hackathon — TigerGraph 2026
GitHub: https://github.com/GauravPatil2515/TigerGraph-RAG

Key Features:
    - Multi-mode execution: "quick" (30 queries) or "full" (200 records)
    - Automated data ingestion for TigerGraph and ChromaDB
    - Comprehensive structured logging to console and file
    - Post-run summary and dashboard pointers

Active Pipeline:
    create_schema.py → ingest/tigergraph_ingest.py → pipelines/pipeline_c_graphrag.py
"""

# Force pure-python protobuf implementation to prevent descriptor validation errors
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import argparse
import logging
import sys
from typing import List, Dict, Any, Optional  # FIX: Added Optional

# Ensure project root is in path for local imports
sys.path.insert(0, os.getcwd())
from data.loader import load_pubmedqa
from ingest.chroma_ingest import ingest_to_chroma
from ingest import tigergraph_ingest as tg_client
from pipelines.pipeline_a_raw_llm import RawLLMPipeline
from pipelines.pipeline_b_basic_rag import BasicRAGPipeline
from pipelines.pipeline_c_graphrag import GraphRAGPipeline
from benchmark.queries import BENCHMARK_QUERIES
from benchmark.runner import BenchmarkRunner

# --- Structured Logging Configuration ---
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/benchmark.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the benchmark run.
    
    Returns:
        argparse.Namespace: Parsed arguments with mode and ingest flags.
    """
    parser = argparse.ArgumentParser(description="GraphRAG Hackathon Benchmark Orchestrator")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="quick", 
        choices=["quick", "full"],
        help="Run mode: 'quick' for curated 30 queries, 'full' for 200 dataset records."
    )
    parser.add_argument(
        "--skip-ingest", 
        action="store_true",
        help="Skip the data ingestion phase (useful if DBs are already populated)."
    )
    return parser.parse_args()

def main() -> None:
    """
    Main orchestration logic.
    
    Performs data loading, optional ingestion, pipeline initialization,
    and triggers the benchmark runner.
    """
    args: argparse.Namespace = parse_args()
    logger.info("="*80)
    logger.info(f"🚀 GraphRAG Audit Run | Mode: {args.mode} | Skip Ingest: {args.skip_ingest}")
    logger.info("="*80)

    # 1. Load PubMedQA Dataset
    records: List[Dict[str, Any]] = load_pubmedqa(1000)
    
    # 2. Ingestion Phase
    if not args.skip_ingest:
        logger.info("📥 Ingesting into ChromaDB and TigerGraph...")
        ingest_to_chroma(records)
        tg_client.ingest_documents(records[:500])
    else:
        logger.info("⏭️  Skipping ingestion as requested.")
    
    # 3. Initialize Evaluation Pipelines
    pipes: List[Any] = [RawLLMPipeline(), GraphRAGPipeline()]
    
    if args.mode == "quick":
        pipes.insert(1, BasicRAGPipeline())
    
    # 4. Prepare Evaluation Queries
    queries: List[Dict[str, Any]] = []
    if args.mode == "full":
        queries = [
            {"id": f"full_{i}", "query": r["question"], "reference": r["answer"], "hop_level": 1} 
            for i, r in enumerate(records[:200])
        ]
        logger.info(f"Prepared {len(queries)} records for full benchmark evaluation.")
    else:
        queries = BENCHMARK_QUERIES
        logger.info(f"Using curated benchmark set: {len(queries)} queries.")

    # 5. Execute Benchmark Run
    runner = BenchmarkRunner(pipelines=pipes)
    runner.run(queries, delay=1.0)
    csv_path: Optional[str] = runner.save()
    
    if csv_path:
        logger.info("="*80)
        logger.info(f"✅ Benchmark Complete! Results: {csv_path}")
        logger.info("📊 View performance insights: streamlit run dashboard/app.py")
        logger.info("="*80)

if __name__ == "__main__":
    main()
