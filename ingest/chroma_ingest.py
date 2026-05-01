"""ChromaDB ingestion module for PubMedQA dataset.

Handles ingestion of PubMedQA records into ChromaDB for Basic RAG pipeline.
"""

import logging
from typing import List, Dict, Any
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHROMA_PATH, EMBED_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_collection() -> chromadb.Collection:
    """Get or create the pubmedqa ChromaDB collection.
    
    Returns:
        ChromaDB collection instance
    """
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL, device="cpu")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="pubmedqa",
        embedding_function=ef
    )
    return collection


def ingest_to_chroma(records: List[Dict[str, Any]], batch_size: int = 100) -> None:
    """Ingest PubMedQA records into ChromaDB.
    
    Args:
        records: List of dicts with question, answer, context
        batch_size: Number of records to ingest per batch (default: 100)
    """
    collection = get_collection()
    
    # Check if collection already has documents
    existing_count = collection.count()
    if existing_count > 0:
        logger.info(f"Collection 'pubmedqa' already has {existing_count} documents. Skipping ingestion.")
        return
    
    logger.info(f"Ingesting {len(records)} records into ChromaDB...")
    
    total = len(records)
    for i in range(0, total, batch_size):
        batch = records[i:i + batch_size]
        
        ids = [f"pubmed_{i + j}" for j in range(len(batch))]
        documents = []
        metadatas = []
        
        for record in batch:
            # Document = question + " " + context (first 500 chars)
            context_preview = record["context"][:500] if record["context"] else ""
            doc_text = f"{record['question']} {context_preview}"
            documents.append(doc_text)
            
            # Metadata
            metadatas.append({
                "question": record["question"],
                "answer": record["answer"],
                "context_preview": context_preview
            })
        
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"  Ingested batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ({i+len(batch)}/{total})")
    
    final_count = collection.count()
    logger.info(f"Ingestion complete. Collection now has {final_count} documents.")


if __name__ == "__main__":
    # Test with sample data
    from data.loader import load_pubmedqa
    
    logger.info("Testing ChromaDB ingestion...")
    records = load_pubmedqa(100)  # Load just 100 for testing
    ingest_to_chroma(records)
    
    # Verify
    collection = get_collection()
    logger.info(f"Collection count: {collection.count()}")
    
    # Test query
    results = collection.query(
        query_texts=["What are the symptoms of diabetes?"],
        n_results=3
    )
    logger.info(f"\nQuery test results:")
    for i, doc in enumerate(results["documents"][0]):
        logger.info(f"  Result {i+1}: {doc[:100]}...")
