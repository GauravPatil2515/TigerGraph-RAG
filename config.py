"""Central configuration loader for GraphRAG Hackathon.

This module loads all environment variables and exports them as module-level
constants for use across the application.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Groq LLM Configuration
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.3-70b-versatile"
COST_PER_1K_TOKENS: float = 0.00059

# TigerGraph Configuration
TG_HOST: str = os.getenv("TIGERGRAPH_HOST", "")
TG_GRAPHNAME: str = os.getenv("TIGERGRAPH_GRAPHNAME", "MyGraph")
TG_USERNAME: str = os.getenv("TIGERGRAPH_USERNAME", "tigergraph")
TG_PASSWORD: str = os.getenv("TIGERGRAPH_PASSWORD", "")
TG_SECRET: str = os.getenv("TIGERGRAPH_SECRET", "")

# ChromaDB Configuration - absolute path (force absolute)
_chroma_path = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_PATH: str = _chroma_path if os.path.isabs(_chroma_path) else os.path.join(BASE_DIR, _chroma_path.replace("./", ""))

# Results Configuration - absolute path (force absolute)
_results_path = os.getenv("RESULTS_PATH", "./results")
RESULTS_PATH: str = _results_path if os.path.isabs(_results_path) else os.path.join(BASE_DIR, _results_path.replace("./", ""))

# Embedding Model
EMBED_MODEL: str = "all-MiniLM-L6-v2"
