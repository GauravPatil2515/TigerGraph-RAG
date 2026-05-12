"""
Module: config.py
Description: Central configuration loader for the GraphRAG Hackathon project.
             Handles environment variable loading, path management, and
             service credentials for Groq, TigerGraph, and ChromaDB.

Author: Gaurav Patil
Project: GraphRAG Inference Hackathon — TigerGraph 2026
GitHub: https://github.com/GauravPatil2515/TigerGraph-RAG

Key Configuration Variables:
    - GROQ_API_KEY: Authentication for Llama-3 inference
    - TG_HOST: TigerGraph Cloud instance URL
    - TG_SECRET: TigerGraph REST++ authentication secret
    - CHROMA_PATH: Local storage path for vector embeddings
    - COST_PER_1K: Token cost basis for ROI calculations
"""

import os
import urllib3
from dotenv import load_dotenv

# Suppress SSL verification warnings from TigerGraph REST++ calls
# (verify=False is used for TigerGraph Cloud compatibility)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load variables from .env file into environment
load_dotenv()

# Base directory (project root)
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

# --- Groq LLM Configuration ---
GROQ_API_KEY: str   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str     = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
COST_PER_1K: float  = 0.00059

# --- TigerGraph Configuration ---
TG_HOST: str      = os.getenv("TIGERGRAPH_HOST", "")
TG_GRAPHNAME: str = os.getenv("TIGERGRAPH_GRAPHNAME", "MedGraph")
TG_SECRET: str    = os.getenv("TIGERGRAPH_SECRET", "")

# --- Paths ---
CHROMA_PATH: str   = os.path.join(BASE_DIR, "chroma_db")
RESULTS_PATH: str  = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_PATH, exist_ok=True)

# --- Context truncation policy (single source of truth) ---
MAX_CONTEXT_CHARS: int = 500

# --- Compatibility Aliases ---
COST_PER_1K_TOKENS: float = COST_PER_1K
EMBED_MODEL: str = "all-MiniLM-L6-v2"

def validate_config() -> dict:
    """
    Validate all required environment variables are set.

    Returns:
        dict: {variable_name: is_valid} for each required config.

    Example:
        issues = {k: v for k, v in validate_config().items() if not v}
        if issues:
            raise ValueError(f"Missing config: {list(issues.keys())}")
    """
    return {
        "GROQ_API_KEY":  bool(GROQ_API_KEY),
        "TG_HOST":       bool(TG_HOST and "your-instance" not in TG_HOST),
        "TG_SECRET":     bool(TG_SECRET and len(TG_SECRET) > 10),
        "TG_GRAPHNAME":  bool(TG_GRAPHNAME),
        "CHROMA_PATH":   os.path.exists(CHROMA_PATH),
    }
