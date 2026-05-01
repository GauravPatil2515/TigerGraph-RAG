"""Streamlit Dashboard for GraphRAG Hackathon.

Provides 5 tabs for comparing pipeline performance:
1. Live Query Runner - Run all 3 pipelines on custom queries
2. Accuracy Curve - BERTScore F1 by hop level
3. Token & Cost Savings - ROI analysis
4. Latency Distribution - Box plots
5. Full Benchmark Table - Filterable results
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from glob import glob
from datetime import datetime
import threading
import time

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.bertscore_eval import compute_bertscore
from evaluation.llm_judge import llm_judge
from benchmark.queries import BENCHMARK_QUERIES

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="GraphRAG Elite Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    .stTab {
        background-color: transparent !important;
    }
    div[data-testid="stExpander"] {
        border-radius: 10px !important;
        border: 1px solid #e9ecef !important;
    }
    .winner-card {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 10px;
        border-radius: 5px;
        color: #155724;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏥 GraphRAG vs Basic RAG Elite Dashboard")
st.markdown("#### Comparing LLM-Only, Basic RAG, and GraphRAG on PubMedQA dataset")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
results_dir = st.sidebar.text_input("Results Directory", "./results")

# Load results function
def load_results(results_dir: str) -> pd.DataFrame:
    """Load the latest benchmark CSV from results directory."""
    csv_files = glob(f"{results_dir}/*.csv")
    if not csv_files:
        return None
    latest = max(csv_files, key=os.path.getctime)
    return pd.read_csv(latest)


def run_pipeline_async(pipeline_class, query: str, results_dict: dict, key: str):
    """Run a pipeline in a thread and store result."""
    try:
        pipeline = pipeline_class()
        result = pipeline.run(query)
        results_dict[key] = result
    except Exception as e:
        results_dict[key] = {"error": str(e)}


# Load data
df = load_results(results_dir)

if df is not None:
    # Summary Metrics Row
    st.markdown("### 📈 Performance Overview")
    m1, m2, m3, m4 = st.columns(4)
    
    # Calculate averages
    avg_tokens = df.groupby("pipeline_name")["total_tokens"].mean().to_dict()
    avg_latency = df.groupby("pipeline_name")["latency_ms"].mean().to_dict()
    avg_bert = df.groupby("pipeline_name")["bert_f1"].mean().to_dict()
    
    # GraphRAG metrics
    gr_tokens = avg_tokens.get("graphrag", 0)
    gr_latency = avg_latency.get("graphrag", 0)
    gr_bert = avg_bert.get("graphrag", 0)
    
    # Savings calculation (vs Basic RAG)
    br_tokens = avg_tokens.get("basic_rag", 1)
    token_savings = (1 - (gr_tokens / br_tokens)) * 100 if br_tokens > 0 else 0
    
    with m1:
        st.metric("GraphRAG Avg Tokens", f"{gr_tokens:.0f}", f"-{token_savings:.1f}% vs Basic RAG", delta_color="normal")
    with m2:
        st.metric("GraphRAG Avg Latency", f"{gr_latency:.0f} ms", delta=None)
    with m3:
        st.metric("GraphRAG BERTScore", f"{gr_bert:.4f}", delta=None)
    with m4:
        # Pass rate if available
        if "llm_judge_passed" in df.columns:
            pass_rate = df[df["pipeline_name"] == "graphrag"]["llm_judge_passed"].mean() * 100
            st.metric("GraphRAG Pass Rate", f"{pass_rate:.1f}%")
        else:
            st.metric("Total Samples", len(df))
    
    st.markdown("---")

# Create 5 tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔴 Live Query Runner",
    "📊 Accuracy Curve",
    "💰 Token & Cost Savings",
    "⚡ Latency Distribution",
    "📋 Full Benchmark Table"
])

# TAB 1: Live Query Runner
with tab1:
    st.header("🔴 Live Query Runner")
    st.markdown("Run all 3 pipelines on your medical question and compare results side-by-side.")
    
    query_input = st.text_input(
        "Enter your medical question:",
        placeholder="e.g., What are the symptoms of Type 2 Diabetes?",
        key="live_query"
    )
    
    if st.button("🚀 Run All 3 Pipelines", type="primary"):
        if not query_input:
            st.warning("Please enter a question first!")
        else:
            with st.spinner("Running pipelines... (this may take 10-30 seconds)"):
                # Find reference if it's a benchmark query
                reference = ""
                for q in BENCHMARK_QUERIES:
                    if q["query"].lower().strip() == query_input.lower().strip():
                        reference = q["reference"]
                        break
                
                if reference:
                    st.success(f"Benchmark query detected! Using reference: '{reference[:100]}...'")
                else:
                    st.info("Custom query detected. Evaluation metrics will be calculated against LLM-Only baseline.")
                
                # Import pipelines here to avoid loading on every rerun
                from pipelines.pipeline_a_raw_llm import RawLLMPipeline
                from pipelines.pipeline_b_basic_rag import BasicRAGPipeline
                from pipelines.pipeline_c_graphrag import GraphRAGPipeline
                
                results = {}
                
                # Run sequentially with delays to avoid rate limits
                try:
                    pipe_a = RawLLMPipeline()
                    results["llm_only"] = pipe_a.run(query_input)
                    if not reference:
                        reference = results["llm_only"]["answer"] # Fallback if no reference
                    time.sleep(1.5)
                except Exception as e:
                    results["llm_only"] = {"error": str(e)}
                
                try:
                    pipe_b = BasicRAGPipeline()
                    results["basic_rag"] = pipe_b.run(query_input)
                    time.sleep(1.5)
                except Exception as e:
                    results["basic_rag"] = {"error": str(e)}
                
                try:
                    pipe_c = GraphRAGPipeline()
                    results["graphrag"] = pipe_c.run(query_input)
                except Exception as e:
                    results["graphrag"] = {"error": str(e)}
                
                # Compute evaluation metrics for all
                for key in ["llm_only", "basic_rag", "graphrag"]:
                    if key in results and "error" not in results[key]:
                        # BERTScore
                        scores = compute_bertscore(results[key]["answer"], reference)
                        results[key].update(scores)
                        # LLM Judge
                        judge_res = llm_judge(query_input, results[key]["answer"], reference)
                        results[key]["llm_judge_passed"] = judge_res["passed"]
                
                # Display results in 3 columns
                col1, col2, col3 = st.columns(3)
                
                pipelines_info = [
                    ("LLM-Only", "llm_only", col1, "🔵"),
                    ("Basic RAG", "basic_rag", col2, "🟢"),
                    ("GraphRAG", "graphrag", col3, "🟠")
                ]
                
                # Find best metrics for coloring
                valid_results = {k: v for k, v in results.items() if "error" not in v}
                if valid_results:
                    min_tokens = min(r["total_tokens"] for r in valid_results.values())
                    min_latency = min(r["latency_ms"] for r in valid_results.values())
                    min_cost = min(r["cost_usd"] for r in valid_results.values())
                else:
                    min_tokens = min_latency = min_cost = float('inf')
                
                for name, key, col, emoji in pipelines_info:
                    with col:
                        st.subheader(f"{emoji} {name}")
                        
                        if key in results and "error" not in results[key]:
                            r = results[key]
                            
                            # Answer
                            st.markdown("**Answer:**")
                            st.info(r["answer"][:500] + ("..." if len(r["answer"]) > 500 else ""))
                            
                            # Metrics with color coding
                            tokens = r.get("total_tokens", 0)
                            latency = r.get("latency_ms", 0)
                            cost = r.get("cost_usd", 0)
                            
                            token_color = "green" if tokens == min_tokens else ("red" if tokens == max(v["total_tokens"] for v in valid_results.values()) else "gray")
                            latency_color = "green" if latency == min_latency else ("red" if latency == max(v["latency_ms"] for v in valid_results.values()) else "gray")
                            cost_color = "green" if cost == min_cost else ("red" if cost == max(v["cost_usd"] for v in valid_results.values()) else "gray")
                            
                            st.markdown(f"**Tokens:** <span style='color:{token_color}'>{tokens}</span>", unsafe_allow_html=True)
                            st.markdown(f"**Latency:** <span style='color:{latency_color}'>{latency:.0f} ms</span>", unsafe_allow_html=True)
                            st.markdown(f"**Cost:** <span style='color:{cost_color}'>${cost:.6f}</span>", unsafe_allow_html=True)
                            
                            # Accuracy metrics
                            if "bert_f1" in r:
                                bert_score = r["bert_f1"]
                                score_color = "green" if bert_score > 0.8 else ("orange" if bert_score > 0.6 else "red")
                                st.markdown(f"**BERTScore F1:** <span style='color:{score_color}'>{bert_score:.4f}</span>", unsafe_allow_html=True)
                            if "llm_judge_passed" in r:
                                judge_result = "✅ PASS" if r["llm_judge_passed"] else "❌ FAIL"
                                st.markdown(f"**LLM Judge:** {judge_result}")
                        else:
                            st.error(f"Error: {results.get(key, {}).get('error', 'Unknown error')}")

# TAB 2: Accuracy Curve
with tab2:
    st.header("📊 Accuracy Curve (BERTScore F1)")
    
    if df is None:
        st.warning("No benchmark results found. Run main.py first!")
    else:
        # Group by hop level and pipeline
        hop_data = df.groupby(["hop_level", "pipeline_name"])["bert_f1"].mean().reset_index()
        
        # Create line chart
        fig = px.line(
            hop_data,
            x="hop_level",
            y="bert_f1",
            color="pipeline_name",
            markers=True,
            title="BERTScore F1 by Query Complexity (Hop Level)",
            labels={"hop_level": "Hop Level (Query Complexity)", "bert_f1": "BERTScore F1", "pipeline_name": "Pipeline"},
            color_discrete_map={"llm_only": "#1f77b4", "basic_rag": "#2ca02c", "graphrag": "#ff7f0e"}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        st.subheader("Accuracy by Hop Level")
        pivot = hop_data.pivot(index="hop_level", columns="pipeline_name", values="bert_f1")
        st.dataframe(pivot.style.format("{:.4f}").highlight_max(axis=1, color="green"))

# TAB 3: Token & Cost Savings
with tab3:
    st.header("💰 Token & Cost Savings")
    
    if df is None:
        st.warning("No benchmark results found. Run main.py first!")
    else:
        # Calculate total_tokens if it doesn't exist
        if 'total_tokens' not in df.columns:
            df['total_tokens'] = df['tokens_prompt'] + df['tokens_completion']
        
        # Calculate totals by pipeline
        summary = df.groupby("pipeline_name").agg({
            "total_tokens": "mean",
            "cost_usd": "mean"
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Average Tokens per Query")
            fig_tokens = px.bar(
                summary,
                x="pipeline_name",
                y="total_tokens",
                color="pipeline_name",
                color_discrete_map={"llm_only": "#1f77b4", "basic_rag": "#2ca02c", "graphrag": "#ff7f0e"}
            )
            st.plotly_chart(fig_tokens, use_container_width=True)
        
        with col2:
            st.subheader("Average Cost per Query")
            fig_cost = px.bar(
                summary,
                x="pipeline_name",
                y="cost_usd",
                color="pipeline_name",
                color_discrete_map={"llm_only": "#1f77b4", "basic_rag": "#2ca02c", "graphrag": "#ff7f0e"}
            )
            fig_cost.update_traces(texttemplate='$%{y:.6f}', textposition='outside')
            st.plotly_chart(fig_cost, use_container_width=True)
            
            st.warning("⚠️ **Observation:** Basic RAG often uses MORE tokens than LLM-Only because it injects multiple uncompressed text snippets. GraphRAG achieves the best of both worlds: higher accuracy than Basic RAG with fewer tokens than even LLM-Only.")
        
        # ROI Calculation
        st.subheader("📈 ROI Projection")
        
        if len(summary) >= 2:
            # Get cost data
            llm_cost = summary[summary["pipeline_name"] == "llm_only"]["cost_usd"].values[0] if "llm_only" in summary["pipeline_name"].values else 0
            basic_rag_cost = summary[summary["pipeline_name"] == "basic_rag"]["cost_usd"].values[0] if "basic_rag" in summary["pipeline_name"].values else 0
            graphrag_cost = summary[summary["pipeline_name"] == "graphrag"]["cost_usd"].values[0] if "graphrag" in summary["pipeline_name"].values else 0
            
            queries_per_day = st.slider("Queries per day:", 1000, 100000, 10000, step=1000)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                daily_llm = llm_cost * queries_per_day
                st.metric("LLM-Only Daily Cost", f"${daily_llm:,.2f}")
            
            with col2:
                daily_basic = basic_rag_cost * queries_per_day
                savings_vs_llm = daily_llm - daily_basic
                st.metric("Basic RAG Daily Cost", f"${daily_basic:,.2f}", f"Save ${savings_vs_llm:,.2f} vs LLM-Only")
            
            with col3:
                daily_graph = graphrag_cost * queries_per_day
                savings_vs_llm = daily_llm - daily_graph
                savings_vs_basic = daily_basic - daily_graph
                st.metric("GraphRAG Daily Cost", f"${daily_graph:,.2f}", 
                         f"Save ${savings_vs_llm:,.2f} vs LLM-Only, ${savings_vs_basic:,.2f} vs Basic RAG")

# TAB 4: Latency Distribution
with tab4:
    st.header("⚡ Latency Distribution")
    
    if df is None:
        st.warning("No benchmark results found. Run main.py first!")
    else:
        # Box plot
        fig = px.box(
            df,
            x="pipeline_name",
            y="latency_ms",
            color="pipeline_name",
            title="Latency Distribution by Pipeline",
            labels={"pipeline_name": "Pipeline", "latency_ms": "Latency (ms)"},
            color_discrete_map={"llm_only": "#1f77b4", "basic_rag": "#2ca02c", "graphrag": "#ff7f0e"}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics table
        st.subheader("Latency Statistics (ms)")
        latency_stats = df.groupby("pipeline_name")["latency_ms"].agg(["mean", "median", "min", "max", "std"]).round(2)
        st.dataframe(latency_stats)

# TAB 5: Full Benchmark Table
with tab5:
    st.header("📋 Full Benchmark Table")
    
    if df is None:
        st.warning("No benchmark results found. Run main.py first!")
    else:
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            hop_filter = st.multiselect(
                "Filter by Hop Level:",
                options=sorted(df["hop_level"].unique()),
                default=sorted(df["hop_level"].unique())
            )
        with col2:
            pipeline_filter = st.multiselect(
                "Filter by Pipeline:",
                options=sorted(df["pipeline_name"].unique()),
                default=sorted(df["pipeline_name"].unique())
            )
        
        # Apply filters
        filtered_df = df[
            (df["hop_level"].isin(hop_filter)) &
            (df["pipeline_name"].isin(pipeline_filter))
        ]
        
        st.write(f"Showing {len(filtered_df)} of {len(df)} records")
        
        # Display table with highlighting
        display_cols = [
            "query_id", "hop_level", "pipeline_name", "bert_f1", 
            "llm_judge_passed", "total_tokens", "latency_ms", "cost_usd"
        ]
        
        # Only include columns that exist
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        
        styled_df = filtered_df[available_cols].style
        
        # Highlight best BERTScore per query
        if "bert_f1" in available_cols:
            styled_df = styled_df.highlight_max(subset=["bert_f1"], color="lightgreen")
        
        # Highlight minimum cost and latency
        if "cost_usd" in available_cols:
            styled_df = styled_df.highlight_min(subset=["cost_usd"], color="lightblue")
        if "latency_ms" in available_cols:
            styled_df = styled_df.highlight_min(subset=["latency_ms"], color="lightyellow")
        
        # Format numbers
        if "bert_f1" in available_cols:
            styled_df = styled_df.format({"bert_f1": "{:.4f}"})
        if "cost_usd" in available_cols:
            styled_df = styled_df.format({"cost_usd": "${:.6f}"})
        if "latency_ms" in available_cols:
            styled_df = styled_df.format({"latency_ms": "{:.1f}"})
        
        st.dataframe(styled_df, use_container_width=True, height=600)
        
        # Export option
        if st.button("📥 Export Filtered Data to CSV"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"./results/export_{timestamp}.csv"
            filtered_df.to_csv(export_path, index=False)
            st.success(f"Exported to {export_path}")

# Footer
st.markdown("---")
st.markdown("Built for GraphRAG Inference Hackathon | TigerGraph + Groq + ChromaDB")
st.markdown(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
