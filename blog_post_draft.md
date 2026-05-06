# How We Cut LLM Token Costs by 79.5% Using TigerGraph GraphRAG

## The Problem: The "RAG Token Paradox"
Standard Vector RAG often retrieves large chunks of text that contain mostly noise. This leads to:
1. **High Costs**: Paying for tokens that don't help answer the question.
2. **Latency**: LLMs take longer to process bloated contexts.
3. **Accuracy Issues**: Irrelevant context can distract the model.

## The Solution: Knowledge Graph-Augmented Retrieval
By using **TigerGraph Cloud**, we built a medical knowledge graph from the **PubMedQA** dataset. Instead of searching by similarity alone, we perform multi-hop traversals:
`Question -> Entity -> Related Documents -> Concise Context`.

## The Numbers That Matter

| Pipeline    | Tokens | Cost/query | BERTScore |
|-------------|--------|------------|-----------|
| LLM-Only    | ~150   | $0.000088  | baseline  |
| Basic RAG   | ~610   | $0.000360  | 0.87      |
| **GraphRAG**| **~125**| **$0.000074**| **0.93** |

**Result: 79.5% fewer tokens. Same accuracy. Significantly more efficient.**

## Technical Stack
- **Database**: TigerGraph Cloud (MedGraph)
- **LLM**: Llama 3.3 70B (via Groq)
- **Framework**: Python, pyTigerGraph, ChromaDB (for baseline)
- **Evaluation**: BERTScore, LLM-as-a-Judge

## ROI Analysis
At 100,000 queries/day, switching from Basic RAG to GraphRAG saves over **$10,000/year** while maintaining superior medical precision.

---
*Stay tuned for the full technical walkthrough and GitHub repository link!*
