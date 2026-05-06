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
from dotenv import load_dotenv

# Load variables from .env file into environment
load_dotenv()

# Base directory (project root) - used for absolute path construction
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

# --- Groq LLM Configuration ---
# API Key for accessing Groq's high-speed inference engine
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
# Model identifier for the primary LLM
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# Estimated cost per 1,000 tokens for ROI and savings metrics
COST_PER_1K: float = 0.00059

# --- TigerGraph Configuration ---
# Host URL for the TigerGraph Cloud instance
TG_HOST: str = os.getenv("TIGERGRAPH_HOST", "")
# Name of the graph schema used for MedGraph
TG_GRAPHNAME: str = os.getenv("TIGERGRAPH_GRAPHNAME", "MedGraph")
# Secret key for generating REST++ authentication tokens
TG_SECRET: str = os.getenv("TIGERGRAPH_SECRET", "")

# --- Paths ---
# Directory for ChromaDB vector storage
CHROMA_PATH: str = os.path.join(BASE_DIR, "chroma_db")
# Directory for benchmark results and CSV exports
RESULTS_PATH: str = os.path.join(BASE_DIR, "results")

# Ensure required directories exist for file writing
os.makedirs(RESULTS_PATH, exist_ok=True)

# --- Compatibility Aliases ---
COST_PER_1K_TOKENS: float = COST_PER_1K
EMBED_MODEL: str = "all-MiniLM-L6-v2"

def validate_config() -> dict[str, bool]:
    """
    Validate all required environment variables are set.
    
    Checks for the existence and basic validity of keys required
    for the pipelines to function correctly.
    
    Returns:
        dict: {variable_name: is_valid} for each required config.
        
    Example:
        issues = {k: v for k, v in validate_config().items() if not v}
        if issues:
            raise ValueError(f"Missing config: {list(issues.keys())}")
    """
    return {
        "GROQ_API_KEY":     bool(GROQ_API_KEY),
        "TG_HOST":          bool(TG_HOST and "your-instance" not in TG_HOST),
        "TG_SECRET":        bool(TG_SECRET and len(TG_SECRET) > 10),
        "TG_GRAPHNAME":     bool(TG_GRAPHNAME),
        "CHROMA_PATH":      os.path.exists(CHROMA_PATH) if os.path.exists(CHROMA_PATH) else False,
    }
