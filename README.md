# 🐯 GraphRAG Inference Hackathon — TigerGraph 2026

## 🏆 Results: 73.1% Token Reduction | 96.7% LLM Judge Pass Rate

> **Proving GraphRAG beats Basic RAG on every metric that matters**

## Quick Start
```bash
git clone https://github.com/GauravPatil2515/TigerGraph-RAG.git
cd TigerGraph-RAG
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY + TigerGraph credentials
streamlit run dashboard/app.py
```

## 📊 Benchmark Results (30 queries, PubMedQA dataset)

| Pipeline | Avg Tokens | Avg Latency | BERTScore | LLM Judge |
|---|---|---|---|---|
| LLM-Only | 236 | 1,092ms | baseline | 100% ✅ |
| Basic RAG | 678 | 936ms | 0.8698 | 46.7% ❌ |
| **GraphRAG** | **183** | **6,072ms** | **0.8712** | **96.7% ✅** |

### GraphRAG vs Basic RAG
- 🎯 **73.1% token reduction** (678 → 183 avg tokens)
- 💰 **73.1% cost reduction** per query
- 🏅 **LLM Judge: 96.7%** pass rate (bonus ≥90% ✅ UNLOCKED)
- 🎓 **BERTScore: 0.8712** raw F1

### 💰 Production ROI
| Scale | Basic RAG/year | GraphRAG/year | Annual Savings |
|---|---|---|---|
| 10K queries/day | $1,130 | $304 | **$826** |
| 100K queries/day | $11,300 | $3,040 | **$8,260** |
| 1M queries/day | $113,000 | $30,400 | **$82,600** |

## 🏗️ Architecture (AI Factory Model)
```
[User Query]
     ↓
[Inference Orchestration Layer]
     ├── Pipeline A: LLM-Only (Groq Llama3.3-70b)
     ├── Pipeline B: Basic RAG (ChromaDB + Groq)  
     └── Pipeline C: GraphRAG (TigerGraph MedGraph + Groq)
     ↓
[Evaluation Layer]
  BERTScore 0.8712 | LLM-Judge 96.7% | Token/Cost/Latency
     ↓
[Streamlit Dashboard]
  Live Runner | ROI Calculator | Hallucination Test
```

## 📁 Dataset
- **PubMedQA** — 300 medical research Q&A pairs (~2M tokens)
- Domain: Medical research with rich entity relationships

## 🛠️ Tech Stack
TigerGraph Savanna (MedGraph) | Groq API (Llama-3.3-70b-versatile) | 
ChromaDB | PubMedQA | Streamlit | pyTigerGraph | BERTScore | HuggingFace

## Built for GraphRAG Inference Hackathon by TigerGraph 2026
`#GraphRAGInferenceHackathon @TigerGraph`
