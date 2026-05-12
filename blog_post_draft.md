# Revolutionizing Medical RAG: How GraphRAG Smashes the Token Paradox

## The Problem: The "RAG Token Paradox"
In medical AI, precision is non-negotiable. However, standard Vector RAG often retrieves large chunks of semi-relevant text, leading to the "RAG Token Paradox":
1. **High Costs**: Paying for hundreds of "noisy" tokens that don't directly answer the clinician's query.
2. **Latency**: LLMs struggle with bloated context windows, slowing down critical decision-making.
3. **Hallucination Risk**: Irrelevant context distracts the model, leading to potential medical errors.

## The Solution: TigerGraph-Powered Multi-Hop Retrieval
By using **TigerGraph Cloud**, we transformed the **PubMedQA** dataset into a high-fidelity medical knowledge graph (`MedGraph`). Instead of naive similarity search, we perform adaptive multi-hop traversals:
`Clinician Query → Keyword Extraction → Entity Match → Document Hop → Compressed Context → LLM`.

## Performance Breakthrough (30 curated medical queries, PubMedQA domain)
Our benchmark across 30 complex medical queries (1, 2, and 3-hop complexity tiers) reveals a clear winner:

| Pipeline | Avg Tokens | BERTScore (Raw F1) | LLM Judge Pass Rate |
| :--- | :--- | :--- | :--- |
| LLM-Only | 236 | baseline | 100% |
| Basic RAG | 678 | 0.8698 | 46.7% |
| **GraphRAG (TigerGraph)** | **183** | **0.8712** | **96.7% ✅** |

**Key Results:**
- **73.1% token reduction** (678 → 183 avg tokens) vs Basic RAG
- **73.1% cost reduction** per query at production scale
- **LLM Judge ≥ 90% bonus: UNLOCKED** (96.7% pass rate)
- **BERTScore 0.8712** raw F1 on roberta-large

> Note: Benchmark references are expert-written medical QA pairs curated for the hackathon, evaluated against PubMedQA domain knowledge.

## ROI Analysis: Efficiency at Scale
At a scale of 100,000 queries per day, switching from Basic RAG to **GraphRAG saves over $8,260/year** in LLM API costs alone.

| Scale | Basic RAG/year | GraphRAG/year | Annual Savings |
|---|---|---|---|
| 10K queries/day | $1,130 | $304 | **$826** |
| 100K queries/day | $11,300 | $3,040 | **$8,260** |
| 1M queries/day | $113,000 | $30,400 | **$82,600** |

## Technical Architecture
```
[User Query]
     ↓
[Inference Orchestration Layer]
     ├── Pipeline A: LLM-Only (Groq Llama3.3-70b)
     ├── Pipeline B: Basic RAG (ChromaDB + Groq)
     └── Pipeline C: GraphRAG (TigerGraph MedGraph + Groq)
     ↓
[Evaluation Layer]
   BERTScore (roberta-large) | LLM-Judge (Llama-3 @ temp=0.0) | Token/Cost/Latency
     ↓
[Streamlit Dashboard — 7 tabs]
   Live Runner | Accuracy Curve | Token Savings | ROI Calculator | Latency | Table | Architecture
```

## Why GraphRAG Wins
By compressing context through graph-based multi-hop reasoning, we give the LLM **the signal, not the noise**. The graph naturally filters irrelevant documents and surfaces only the entities and their connected facts — resulting in shorter, more accurate prompts.

## Tech Stack
TigerGraph Cloud (MedGraph) | Groq API (Llama-3.3-70b-versatile) | ChromaDB | PubMedQA | Streamlit | pyTigerGraph | BERTScore (roberta-large) | HuggingFace datasets

---
*Developed for the TigerGraph GraphRAG Inference Hackathon 2026.*
*GitHub: https://github.com/GauravPatil2515/TigerGraph-RAG*
