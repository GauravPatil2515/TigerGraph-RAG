"""
Module: chroma_ingest.py
Description: ChromaDB ingestion module for the PubMedQA dataset. 
             Handles persistent storage and vector indexing of medical records
             to support the Basic RAG (Pipeline B) baseline.

Author: Gaurav Patil
Project: GraphRAG Inference Hackathon — TigerGraph 2026

Key Features:
    - Persistent vector storage using ChromaDB
    - Sentence Transformer embeddings (all-MiniLM-L6-v2)
    - Batch ingestion with duplication checks
    - Metadata-rich indexing for efficient retrieval
"""

import logging
from typing import List, Dict, Any
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import sys
import os

# Ensure project root is in path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHROMA_PATH, EMBED_MODEL

# Configure structured logging
logger = logging.getLogger(__name__)

def get_collection() -> chromadb.Collection:
    """
    Retrieve or create the 'pubmedqa' ChromaDB collection.
    
    Initializes the embedding function and persistent client using 
    parameters defined in config.py.

    Returns:
        chromadb.Collection: The initialized ChromaDB collection instance.
        
    Example:
        col = get_collection()
        print(f"Collection contains {col.count()} documents.")
    """
    # Use CPU for embedding generation to ensure compatibility across deployment environments
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL, device="cpu")
    
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="pubmedqa",
        embedding_function=ef
    )
    return collection

def ingest_to_chroma(records: List[Dict[str, Any]], batch_size: int = 100) -> None:
    """
    Ingest a list of PubMedQA records into ChromaDB.
    
    Performs batch upserts to optimize performance. Skips ingestion if the 
    collection is already populated to avoid redundant API calls and storage.

    Args:
        records: List of sanitized dicts containing question, answer, and context.
        batch_size: Number of records to process per database transaction.
        
    Example:
        data = load_pubmedqa(1000)
        ingest_to_chroma(data)
    """
    collection: chromadb.Collection = get_collection()
    
    # Simple check to prevent redundant ingestion
    existing_count: int = collection.count()
    if existing_count > 0:
        logger.info(f"Collection 'pubmedqa' already contains {existing_count} documents. Skipping ingestion.")
        return
    
    logger.info(f"Starting ingestion of {len(records)} records into ChromaDB...")
    
    total: int = len(records)
    for i in range(0, total, batch_size):
        batch: List[Dict[str, Any]] = records[i:i + batch_size]
        
        # Construct unique IDs and document content for the vector engine
        ids: List[str] = [f"pubmed_{i + j}" for j in range(len(batch))]
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        
        for record in batch:
            # Document content is a concatenation of the question and a context preview
            # to provide sufficient semantic signal for similarity search.
            context_preview: str = record["context"][:500] if record["context"] else ""
            doc_text: str = f"{record['question']} {context_preview}"
            documents.append(doc_text)
            
            # Store full QA pairs in metadata for high-fidelity retrieval during the RAG phase
            metadatas.append({
                "question": record["question"],
                "answer": record["answer"],
                "context_preview": context_preview
            })
        
        # Use upsert to handle potential ID collisions gracefully
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"  Ingested batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ({i+len(batch)}/{total})")
    
    final_count: int = collection.count()
    logger.info(f"✅ ChromaDB ingestion complete. Collection now has {final_count} documents.")

if __name__ == "__main__":
    # Integration test for standalone verification
    logging.basicConfig(level=logging.INFO)
    from data.loader import load_pubmedqa
    
    test_records = load_pubmedqa(50)
    ingest_to_chroma(test_records)
