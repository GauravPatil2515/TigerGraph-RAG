"""TigerGraph GraphRAG REST API client for ingestion and querying.

This module provides functions to interact with the TigerGraph GraphRAG REST API
for document ingestion and GraphRAG queries.
"""

import logging
from typing import List, Dict, Any
import requests
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TG_HOST, TG_USERNAME, TG_PASSWORD, TG_GRAPHNAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_health() -> bool:
    """Check TigerGraph GraphRAG health endpoint.
    
    Returns:
        True if health check passes (200 OK), False otherwise
    """
    try:
        url = f"{TG_HOST}/health"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            logger.info("TigerGraph GraphRAG health check: OK")
            return True
        else:
            logger.warning(f"TigerGraph GraphRAG health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"TigerGraph health check error: {e}")
        return False


def ingest_documents(records: List[Dict[str, Any]], batch_size: int = 50) -> Dict[str, Any]:
    """Ingest documents into TigerGraph GraphRAG via REST API.
    
    Args:
        records: List of dicts with id, question, answer, context
        batch_size: Number of documents per batch (default: 50)
        
    Returns:
        Status dict with success flag and message
    """
    logger.info(f"Ingesting {len(records)} documents into TigerGraph GraphRAG...")
    
    total = len(records)
    success_count = 0
    
    for i in range(0, total, batch_size):
        batch = records[i:i + batch_size]
        
        # Build documents payload
        documents = []
        for j, record in enumerate(batch):
            content = f"Question: {record['question']}\nContext: {record['context']}\nAnswer: {record['answer']}"
            doc = {
                "id": f"pubmed_{i + j}",
                "content": content,
                "metadata": {
                    "answer": record["answer"],
                    "question": record["question"]
                }
            }
            documents.append(doc)
        
        payload = {"documents": documents}
        
        try:
            url = f"{TG_HOST}/documents/ingest"
            response = requests.post(
                url,
                json=payload,
                auth=(TG_USERNAME, TG_PASSWORD),
                timeout=60
            )
            
            if response.status_code == 200:
                success_count += len(batch)
                logger.info(f"  Ingested batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ({i+len(batch)}/{total})")
            else:
                logger.error(f"  Batch {i//batch_size + 1} failed: {response.status_code} - {response.text[:200]}")
        except Exception as e:
            logger.error(f"  Batch {i//batch_size + 1} error: {e}")
        
        # Small delay between batches
        time.sleep(0.5)
    
    status = {
        "success": success_count == total,
        "total": total,
        "ingested": success_count,
        "message": f"Ingested {success_count}/{total} documents"
    }
    
    logger.info(status["message"])
    return status


def query_graphrag(question: str, top_k: int = 5, num_hops: int = 2) -> Dict[str, Any]:
    """Run GraphRAG query via TigerGraph REST API.
    
    Args:
        question: The query question
        top_k: Number of top results to return (default: 5)
        num_hops: Number of graph hops for traversal (default: 2)
        
    Returns:
        Full response dict from GraphRAG API
    """
    payload = {
        "query": question,
        "top_k": top_k,
        "num_hops": num_hops
    }
    
    try:
        url = f"{TG_HOST}/query"
        response = requests.post(
            url,
            json=payload,
            auth=(TG_USERNAME, TG_PASSWORD),
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"GraphRAG query failed: {response.status_code} - {response.text[:200]}")
            return {
                "error": f"HTTP {response.status_code}",
                "message": response.text,
                "answer": None,
                "context": ""
            }
    except Exception as e:
        logger.error(f"GraphRAG query error: {e}")
        return {
            "error": str(e),
            "answer": None,
            "context": ""
        }


def check_ingestion_status() -> Dict[str, Any]:
    """Check document ingestion status.
    
    Returns:
        Status dict from the API
    """
    try:
        url = f"{TG_HOST}/documents/status"
        response = requests.get(url, auth=(TG_USERNAME, TG_PASSWORD), timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Test block
    logger.info("Testing TigerGraph GraphRAG client...")
    
    # Test health check
    if check_health():
        logger.info("✓ Health check passed")
    else:
        logger.warning("✗ Health check failed - TigerGraph may not be configured")
    
    # Test with sample data if available
    try:
        from data.loader import load_pubmedqa
        records = load_pubmedqa(50)  # Load just 50 for testing
        result = ingest_documents(records, batch_size=25)
        logger.info(f"Ingestion result: {result}")
        
        # Test query
        query_result = query_graphrag("What are the symptoms of diabetes?")
        logger.info(f"Query result: {query_result}")
    except Exception as e:
        logger.error(f"Test failed: {e}")
