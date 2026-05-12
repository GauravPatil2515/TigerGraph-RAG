# 🐯 TigerGraph-RAG — Architecture Guide

This document clarifies the **active production pipeline**, which files are entry points, and which are deprecated experiments.

---

## ✅ Active Pipeline (Production)

```
Setup (run once):
  create_schema.py          → Creates TigerGraph MedGraph schema
                              Vertices: Document, Entity
                              Edges:    mentions (Document → Entity)

Data Ingestion:
  data/loader.py            → Loads PubMedQA from HuggingFace (1000 records)
  ingest/chroma_ingest.py   → Embeds records into ChromaDB (Pipeline B)
  ingest/tigergraph_ingest.py → Extracts medical entities, builds graph (Pipeline C)

Benchmark Execution:
  main.py                   → Orchestrates full benchmark run
    --mode quick            → 30 curated queries, all 3 pipelines
    --mode full             → 200 PubMedQA records, Pipeline A + C only
    --skip-ingest           → Skip ingestion if DBs already populated

Pipelines:
  pipelines/pipeline_a_raw_llm.py    → LLM-Only (no retrieval, Groq)
  pipelines/pipeline_b_basic_rag.py  → Basic RAG (ChromaDB + Groq)
  pipelines/pipeline_c_graphrag.py   → GraphRAG (TigerGraph 3-hop + Groq)

Evaluation:
  evaluation/bertscore_eval.py → Semantic similarity (roberta-large)
  evaluation/llm_judge.py      → PASS/FAIL factual judgment (Llama-3 @ temp=0.0)
  benchmark/runner.py          → Orchestrates all pipelines × all queries
  benchmark/queries.py         → 30 curated medical queries (10 per hop level)

Dashboard:
  dashboard/app.py            → Streamlit 7-tab dashboard
  streamlit run dashboard/app.py

Diagnostics:
  check_connections.py        → Verify TigerGraph, ChromaDB, Groq connectivity
  python check_connections.py
```

---

## ⚠️ Deprecated / Experimental (Do Not Use at Runtime)

| File | Reason |
|---|---|
| `graph/loader.py` | Alternative spaCy NER-based ingestion. Uses different schema (Disease/Drug/Symptom vertices). Not connected to active pipeline. |
| `graph/schema.py` | Connection helper for the above — unused at runtime. |
| `export_csv.py` | Generates hardcoded demo CSVs for schema illustration only. NOT real PubMedQA data. |

These files are kept for reference but **will not work** with the active `MedGraph` schema (Document + Entity vertices).

---

## 🏗️ TigerGraph Schema (Active)

```
Vertex: Document
  PRIMARY_ID: doc_id (STRING)
  question:   STRING
  answer:     STRING
  context:    STRING
  source:     STRING  (= "PubMedQA")

Vertex: Entity
  PRIMARY_ID: name (STRING)  (e.g. "diabetes", "insulin", "kidney")
  entity_type: STRING        (= "medical_term")

Edge: mentions (DIRECTED)
  FROM: Document
  TO:   Entity
```

---

## 🔁 GraphRAG 3-Hop Traversal

```
[User Query: "How does insulin resistance cause kidney failure?"]
        ↓
[extract_keywords()]  →  ["insulin", "resistance", "kidney", "failure"]
        ↓
[classify_query_complexity()]  →  hop_level = 3  ("how does" trigger)
        ↓
HOP 1: Match keywords against Entity vertex IDs in MedGraph
        →  entity: "insulin", "kidney"
        ↓
HOP 2: Find Document vertices that mention those entities
        →  via REST++: GET /graph/MedGraph/vertices/Document?filter=...
        ↓
HOP 3: Extract question + answer from Document attributes
        →  Compressed context ~100-300 chars
        ↓
[Groq Llama-3.3-70b] ← system + compressed context + user query
        ↓
[Answer + metrics: tokens, latency, cost, graph_hops]
```

---

## 📊 Benchmark Results (30 queries, PubMedQA domain)

| Pipeline | Avg Tokens | Avg Latency | BERTScore | LLM Judge |
|---|---|---|---|---|
| LLM-Only  | 236  | 1,092ms | baseline | 100% ✅ |
| Basic RAG | 678  | 936ms   | 0.8698   | 46.7% ❌ |
| **GraphRAG** | **183** | **6,072ms** | **0.8712** | **96.7% ✅** |

- 73.1% token reduction (GraphRAG vs Basic RAG)
- LLM Judge ≥ 90% bonus: **UNLOCKED**
- Latency trade-off: GraphRAG is ~5s slower (graph traversal cost)

---

## 🛠️ Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your GROQ_API_KEY, TIGERGRAPH_HOST, TIGERGRAPH_SECRET

# 3. Check all connections
python check_connections.py

# 4. Create TigerGraph schema (run once)
python create_schema.py

# 5. Run benchmark
python main.py --mode quick

# 6. View dashboard
streamlit run dashboard/app.py
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `pyTigerGraph` | TigerGraph Cloud REST++ connection |
| `groq` | Llama-3.3-70b inference |
| `chromadb` | Vector database for Basic RAG |
| `bert-score` | Semantic evaluation (roberta-large) |
| `sentence-transformers` | Embeddings for ChromaDB |
| `streamlit` | Dashboard UI |
| `datasets` | PubMedQA loader (HuggingFace) |
| `plotly` | Dashboard charts |
