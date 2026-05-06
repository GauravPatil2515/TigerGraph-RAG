"""
Module: integration_test.py
Description: Scratch script for rapid integration testing of all pipelines 
             and evaluation services. Verifies end-to-end functionality 
             before running full benchmark suites.

Author: Gaurav Patil
Project: GraphRAG Inference Hackathon — TigerGraph 2026
"""

import os, sys, time
sys.path.insert(0, os.getcwd())

from pipelines.pipeline_a_raw_llm import RawLLMPipeline
from pipelines.pipeline_b_basic_rag import BasicRAGPipeline
from pipelines.pipeline_c_graphrag import GraphRAGPipeline
from evaluation.bertscore_eval import compute_bertscore
from evaluation.llm_judge import llm_judge

def run_integration_test() -> None:
    """
    Execute a single-query integration test across all three pipelines.
    
    Validates connectivity, inference, and evaluation scoring in 
    a single synchronous execution pass.
    """
    query: str = "What is types of diabetis"
    reference: str = "Diabetes mellitus is classified into three main types: Type 1 (autoimmune destruction of beta cells), Type 2 (insulin resistance), and Gestational Diabetes (pregnancy-induced)."
    
    print(f"🚀 STARTING INTEGRATION TEST")
    print(f"Query: {query}\n")
    
    # Initialize all pipelines to be tested
    pipelines = [
        ("LLM-ONLY", RawLLMPipeline()),
        ("BASIC RAG", BasicRAGPipeline()),
        ("GRAPHRAG", GraphRAGPipeline())
    ]
    
    for name, pipe in pipelines:
        print(f"--- Testing {name} ---")
        start: float = time.time()
        try:
            # Execute pipeline run
            res = pipe.run(query)
            latency: float = (time.time() - start) * 1000
            
            # Semantic Scorer (BERTScore)
            scores = compute_bertscore(res['answer'], reference)
            
            # Factual Judge (LLM-as-a-Judge)
            judge_res = llm_judge(query, res['answer'], reference)
            
            print(f"Result: {res['answer'][:100]}...")
            print(f"Latency: {latency:.0f}ms")
            print(f"BERTScore Raw: {scores['bert_f1_raw']}")
            print(f"LLM Judge: {'✅ PASS' if judge_res['llm_judge_passed'] else '❌ FAIL'}")
            print(f"Status: ✅ SUCCESS\n")
        except Exception as e:
            print(f"Status: ❌ FAILED - {str(e)}\n")

if __name__ == "__main__":
    run_integration_test()
