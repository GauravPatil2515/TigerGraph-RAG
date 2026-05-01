"""Central configuration loader for GraphRAG Hackathon.

This module loads all environment variables and exports them as module-level
constants for use across the application.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Groq LLM Configuration
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "openai/gpt-oss-120b"
COST_PER_1K_TOKENS: float = 0.00059

# TigerGraph Configuration
TG_HOST: str = os.getenv("TIGERGRAPH_HOST", "")
TG_GRAPHNAME: str = os.getenv("TIGERGRAPH_GRAPHNAME", "MyGraph")
TG_USERNAME: str = os.getenv("TIGERGRAPH_USERNAME", "tigergraph")
TG_PASSWORD: str = os.getenv("TIGERGRAPH_PASSWORD", "")

# ChromaDB Configuration
CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma_db")

# Results Configuration
RESULTS_PATH: str = os.getenv("RESULTS_PATH", "./results")

# Embedding Model
EMBED_MODEL: str = "all-MiniLM-L6-v2"
