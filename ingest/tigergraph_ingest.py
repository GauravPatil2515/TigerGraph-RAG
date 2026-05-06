"""
Module: tigergraph_ingest.py
Description: TigerGraph ingestion module for medical GraphRAG.
             Handles entity extraction and graph construction for 
             multi-hop context retrieval.

Author: Gaurav Patil
Project: GraphRAG Inference Hackathon — TigerGraph 2026

Graph Schema:
    - Vertex: Document (question, answer, context)
    - Vertex: Entity (medical terms)
    - Edge: Document -mentions-> Entity
"""

import os
import re
import sys
import time
import requests
import logging
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from dotenv import load_dotenv
import pyTigerGraph as tg

# Ensure project root is in path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.loader import load_pubmedqa

# Load environment variables
load_dotenv()

# Configure structured logging
logger = logging.getLogger(__name__)

def get_token() -> Tuple[str, str]:
    """
    Retrieve OAuth token and REST++ URL for TigerGraph connection.
    
    Returns:
        Tuple[str, str]: (auth_token, restpp_url)
    """
    secret: str = os.getenv("TIGERGRAPH_SECRET", "")
    host: str = os.getenv("TIGERGRAPH_HOST", "")
    graph: str = os.getenv("TIGERGRAPH_GRAPHNAME", "MedGraph")
    
    conn = tg.TigerGraphConnection(host=host, graphname=graph, gsqlSecret=secret, tgCloud=True)
    token = conn.getToken(secret)
    if isinstance(token, tuple): token = token[0]
    return token, conn.restppUrl

def check_health() -> bool:
    """
    Check if the TigerGraph instance is reachable and healthy.
    
    Returns:
        bool: True if healthy, False otherwise.
    """
    try:
        token, restpp_url = get_token()
        graph: str = os.getenv("TIGERGRAPH_GRAPHNAME", "MedGraph")
        headers: dict = {"Authorization": f"Bearer {token}"}
        
        # Test request to check graph connectivity
        response = requests.get(f"{restpp_url}/echo", headers=headers, verify=False, timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"TigerGraph health check failed: {e}")
        return False

def extract_entities(text: str) -> List[str]:
    """
    Extract key medical entities from text using keyword heuristics.
    
    Args:
        text: Input string to scan for medical terms.
        
    Returns:
        List[str]: Unique list of identified medical entities.
    """
    medical_terms: List[str] = [
        "diabetes", "hypertension", "insulin", "glucose", "cancer", "metformin", 
        "asthma", "obesity", "depression", "arthritis", "kidney", "liver", 
        "heart", "lung", "brain", "blood", "therapy", "treatment", "diagnosis", 
        "symptoms", "medication", "surgery", "clinical", "patient", "disease", 
        "disorder", "cholesterol", "pressure", "immune", "infection", "virus"
    ]
    found: List[str] = []
    text_lower: str = text.lower()
    for term in medical_terms:
        if term in text_lower:
            found.append(term)
    return list(set(found))

def ingest_documents(records: List[Dict[str, Any]], limit: int = 500) -> None:
    """
    Ingest medical documents and entities into TigerGraph.
    
    Constructs the graph by creating Document vertices and linking 
    them to Entity vertices based on keyword mentions.

    Args:
        records: List of PubMedQA records to ingest.
        limit: Maximum number of records to process.
    """
    try:
        token, restpp_url = get_token()
    except Exception as e:
        logger.error(f"Failed to initialize TigerGraph connection: {e}")
        return

    graph: str = os.getenv("TIGERGRAPH_GRAPHNAME", "MedGraph")
    headers: Dict[str, str] = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    logger.info(f"Ingesting {min(limit, len(records))} documents into {graph}...")
    
    docs_upserted: int = 0
    entities_upserted: int = 0
    edges_upserted: int = 0

    for i, rec in enumerate(tqdm(records[:limit])):
        doc_id: str = f"pubmed_{i}"
        question: str = rec.get("question", "")[:500]
        answer: str   = rec.get("answer", "")[:500]
        context: str  = rec.get("context", "")[:300]

        # --- STEP 1: Upsert Document Vertex ---
        payload = {
            "vertices": {
                "Document": {
                    doc_id: {
                        "question": {"value": question}, 
                        "answer": {"value": answer}, 
                        "context": {"value": context}, 
                        "source": {"value": "PubMedQA"}
                    }
                }
            }
        }
        r = requests.post(f"{restpp_url}/graph/{graph}", json=payload, headers=headers, verify=False)
        if r.status_code == 200: 
            docs_upserted += 1

        # --- STEP 2: Extract and Upsert Entity Vertices and Edges ---
        entities: List[str] = extract_entities(question + " " + answer)
        for ent in entities:
            # Upsert Entity Vertex
            ent_payload = {"vertices": {"Entity": {ent: {"entity_type": {"value": "medical_term"}}}}}
            requests.post(f"{restpp_url}/graph/{graph}", json=ent_payload, headers=headers, verify=False)
            
            # Upsert Edge (Document -> mentions -> Entity)
            edge_payload = {"edges": {"Document": {doc_id: {"mentions": {"Entity": {ent: {}}}}}}}
            requests.post(f"{restpp_url}/graph/{graph}", json=edge_payload, headers=headers, verify=False)
            
            entities_upserted += 1
            edges_upserted += 1

    logger.info(f"✅ TigerGraph ingestion complete!")
    logger.info(f"📊 Summary: Docs: {docs_upserted}, Entities: {entities_upserted}, Edges: {edges_upserted}")

if __name__ == "__main__":
    # Ingestion test script
    logging.basicConfig(level=logging.INFO)
    test_records = load_pubmedqa(100)
    ingest_documents(test_records, limit=20)
