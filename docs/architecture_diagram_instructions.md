# Architecture Diagram Instructions

## Create at diagrams.net (draw.io)

**URL:** https://app.diagrams.net/

## Diagram Structure

```
[User Query]
     ↓
[Inference Orchestration Layer]
  ├── Route query to all 3 pipelines
  ├── Manage API calls + fallback
  └── Aggregate results
     ↓
┌──────────┬──────────────┬─────────────────┐
│Pipeline A│  Pipeline B  │   Pipeline C    │
│ LLM-Only │  Basic RAG   │   GraphRAG      │
│  Groq    │ChromaDB+Groq │TigerGraph+Groq  │
└──────────┴──────────────┴─────────────────┘
     ↓
[Evaluation Layer]
  BERTScore 0.9278 | LLM-Judge | Token/Cost/Latency
     ↓
[Streamlit Dashboard]
  Live Runner | ROI Calc | Accuracy Curve | Hallucination Test
```

## Steps to Create

1. Go to https://app.diagrams.net/
2. Create New Diagram → Blank
3. Insert shapes:
   - **Top:** Rectangle labeled "👤 USER QUERY"
   - **Middle:** Rectangle labeled "⚙️ INFERENCE ORCHESTRATION LAYER"
   - **3 Boxes below:** 
     - "Pipeline A: LLM-Only (Groq)" - Blue
     - "Pipeline B: Basic RAG (ChromaDB + Groq)" - Green  
     - "Pipeline C: GraphRAG (TigerGraph + Groq)" - Orange
   - **Below:** Rectangle labeled "📊 EVALUATION LAYER"
   - **Bottom:** Rectangle labeled "📈 STREAMLIT DASHBOARD"
4. Connect with arrows
5. Add metrics text:
   - "BERTScore: 0.9278"
   - "Token Reduction: 69.6%"
   - "Cost Reduction: 69.6%"
6. Export as PNG: File → Export As → PNG → Border: 10 → Crop: Diagram
7. Save to `docs/architecture.png`

## Color Scheme

- User Query: #1f77b4 (Blue)
- Orchestration: #ff7f0e (Orange)
- Pipelines: #2ca02c (Green)
- Evaluation: #d62728 (Red)
- Dashboard: #9467bd (Purple)

## Export Settings

- Format: PNG
- Border: 10
- Crop: Diagram (not page)
- Shadows: No
- Grid: No
