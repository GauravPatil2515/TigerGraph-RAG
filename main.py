"""Main entrypoint for GraphRAG Hackathon.

Full pipeline orchestration:
1. Load PubMedQA dataset
2. Ingest into ChromaDB (if needed)
3. Check TigerGraph health and ingest (if healthy)
4. Initialize all 3 pipelines
5. Run benchmark on all 30 queries
6. Save results and print summary
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader import load_pubmedqa
from ingest.chroma_ingest import ingest_to_chroma
from ingest import tigergraph_ingest as tg_client
from pipelines.pipeline_a_raw_llm import RawLLMPipeline
from pipelines.pipeline_b_basic_rag import BasicRAGPipeline
from pipelines.pipeline_c_graphrag import GraphRAGPipeline
from benchmark.queries import BENCHMARK_QUERIES
from benchmark.runner import BenchmarkRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GraphRAG Hackathon - Run benchmark across 3 pipelines"
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip data ingestion (use existing DBs)"
    )
    parser.add_argument(
        "--pipelines",
        type=str,
        default="a,b,c",
        help="Comma-separated list of pipelines to run: a,b,c (default: all)"
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=30,
        help="Number of queries to run (default: 30)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        choices=["quick", "full"],
        help="Benchmark mode: quick (30 queries) or full (200 PubMedQA records)"
    )
    return parser.parse_args()


def main():
    """Main entrypoint."""
    args = parse_args()
    
    logger.info("="*80)
    logger.info("GraphRAG Hackathon - Benchmark Runner")
    logger.info("="*80)
    
    # Parse pipelines to run
    pipeline_map = {"a": "llm_only", "b": "basic_rag", "c": "graphrag"}
    selected_pipelines = [pipeline_map[p.strip()] for p in args.pipelines.split(",") if p.strip() in pipeline_map]
    logger.info(f"Selected pipelines: {selected_pipelines}")
    logger.info(f"Benchmark mode: {args.mode}")
    
    # Limit queries based on mode
    if args.mode == "full":
        queries_to_run = []  # Will use PubMedQA records directly in full mode
        logger.info("Full mode: Will run on 200 PubMedQA records")
    else:
        queries_to_run = BENCHMARK_QUERIES[:args.queries]
        logger.info(f"Quick mode: Running {len(queries_to_run)} benchmark queries")
    
    # 1. Load PubMedQA dataset
    logger.info("\n[Step 1/5] Loading PubMedQA dataset...")
    records = load_pubmedqa(1000)
    
    # 2. Check ChromaDB and ingest if needed
    if not args.skip_ingest:
        logger.info("\n[Step 2/5] Ingesting into ChromaDB...")
        ingest_to_chroma(records)
    else:
        logger.info("\n[Step 2/5] Skipping ChromaDB ingestion (--skip-ingest)")
    
    # 3. Check TigerGraph health and ingest if healthy
    if not args.skip_ingest and "graphrag" in selected_pipelines:
        logger.info("\n[Step 3/5] Checking TigerGraph GraphRAG health...")
        if tg_client.check_health():
            logger.info("TigerGraph is healthy. Ingesting documents...")
            tg_client.ingest_documents(records[:500])  # Use subset for graph
        else:
            logger.warning("TigerGraph health check failed. GraphRAG pipeline may not work.")
    else:
        if args.skip_ingest:
            logger.info("\n[Step 3/5] Skipping TigerGraph ingestion (--skip-ingest)")
        else:
            logger.info("\n[Step 3/5] Skipping TigerGraph (not selected)")
    
    # 4. Initialize selected pipelines
    logger.info("\n[Step 4/5] Initializing pipelines...")
    pipelines = []
    
    if "llm_only" in selected_pipelines:
        pipe_a = RawLLMPipeline()
        pipelines.append(pipe_a)
        logger.info(f"  ✓ Initialized {pipe_a}")
    
    if "basic_rag" in selected_pipelines:
        pipe_b = BasicRAGPipeline()
        pipelines.append(pipe_b)
        logger.info(f"  ✓ Initialized {pipe_b}")
    
    if "graphrag" in selected_pipelines:
        pipe_c = GraphRAGPipeline()
        pipelines.append(pipe_c)
        logger.info(f"  ✓ Initialized {pipe_c}")
    
    if not pipelines:
        logger.error("No valid pipelines selected! Exiting.")
        sys.exit(1)
    
    # 5. Run benchmark (quick or full mode)
    logger.info("\n[Step 5/5] Running benchmark...")
    runner = BenchmarkRunner(pipelines=pipelines, results_dir="./results", mode=args.mode)
    
    if args.mode == "full":
        # Full mode: Run on 200 PubMedQA records with BERTScore only
        summary = runner.run_full_benchmark(records, pipelines, delay_seconds=0.5)
        df = runner.save_csv()
        if summary:
            logger.info("\n" + "="*80)
            logger.info("FULL BENCHMARK SUMMARY")
            logger.info("="*80)
            for pipeline_name, metrics in summary.items():
                logger.info(f"\n  {pipeline_name}:")
                for metric, value in metrics.items():
                    logger.info(f"    {metric}: {value:.4f}")
    else:
        # Quick mode: Run on benchmark queries with full evaluation
        runner.run(queries_to_run, delay_seconds=1.5)
        df = runner.save_csv()
        runner.print_summary(df)
    
    # 7. Final message
    logger.info("\n" + "="*80)
    logger.info("✅ Run complete!")
    logger.info("="*80)
    logger.info(f"Results saved to: {runner.csv_path}")
    logger.info("\n📊 Launch dashboard:")
    logger.info("   streamlit run dashboard/app.py")
    logger.info("="*80)


if __name__ == "__main__":
    main()

