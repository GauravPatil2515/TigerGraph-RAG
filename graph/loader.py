"""
Module: loader.py
Description: Advanced knowledge graph loader using NLP-driven entity extraction.
             Uses spaCy NER to identify medical entities and populate 
             the TigerGraph schema with structured relationships.

Author: Gaurav Patil
Project: GraphRAG Inference Hackathon — TigerGraph 2026
"""

import spacy
from tqdm import tqdm
from graph.schema import get_connection
from typing import List, Tuple, Any

# Load spaCy model for Named Entity Recognition (NER)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # Fallback if model is not downloaded
    nlp = None

def extract_entities(text: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Extract medical entities using spaCy NER.
    
    Identifies diseases, drugs, and symptoms from raw text blocks.

    Args:
        text: Input medical text.
        
    Returns:
        Tuple: (diseases, drugs, symptoms) lists.
    """
    if not nlp:
        return [], [], []
        
    doc = nlp(text)
    diseases: List[str] = [ent.text for ent in doc.ents if ent.label_ in ["DISEASE", "CONDITION"]]
    drugs: List[str]    = [ent.text for ent in doc.ents if ent.label_ in ["CHEMICAL", "DRUG"]]
    symptoms: List[str] = [ent.text for ent in doc.ents if ent.label_ in ["SYMPTOM", "SIGN"]]
    return diseases, drugs, symptoms

def create_schema(conn: Any) -> None:
    """
    Create graph schema for the medical knowledge graph.
    
    Defines vertex types (Disease, Drug, Symptom, Question) and 
    the connecting edge relationships.

    Args:
        conn: Authenticated TigerGraph connection.
    """
    try:
        # Create vertex types
        conn.gsql("CREATE VERTEX Disease (PRIMARY_ID id STRING, name STRING)")
        conn.gsql("CREATE VERTEX Drug (PRIMARY_ID id STRING, name STRING)")
        conn.gsql("CREATE VERTEX Symptom (PRIMARY_ID id STRING, name STRING)")
        conn.gsql("CREATE VERTEX Question (PRIMARY_ID id STRING, text STRING, answer STRING)")
        
        # Create edge types
        conn.gsql("CREATE EDGE has_symptom (FROM Disease, TO Symptom)")
        conn.gsql("CREATE EDGE treated_by (FROM Disease, TO Drug)")
        conn.gsql("CREATE EDGE has_side_effect (FROM Drug, TO Symptom)")
        conn.gsql("CREATE EDGE mentions (FROM Question, TO Disease)")
        conn.gsql("CREATE EDGE mentions_drug (FROM Question, TO Drug)")
        conn.gsql("CREATE EDGE mentions_symptom (FROM Question, TO Symptom)")
        
        print("✅ Schema created successfully!")
    except Exception as e:
        print(f"⚠️ Schema check info: {e}")

def ingest_to_tigergraph(records: List[Dict[str, Any]], batch_size: int = 100) -> Any:
    """
    Ingest data into TigerGraph as a structured knowledge graph.
    
    Extracts entities from each record and builds a multi-hop graph
    connecting questions to their underlying medical concepts.

    Args:
        records: List of medical QA records.
        batch_size: Ingestion batch size (default: 100).
        
    Returns:
        conn: The TigerGraph connection used for ingestion.
    """
    conn = get_connection()
    
    # Ensure schema exists before starting ingestion
    create_schema(conn)
    
    print(f"🚀 Ingesting {len(records)} records into TigerGraph...")
    
    for i, record in enumerate(tqdm(records)):
        qid: str = f"q_{i}"
        question_text: str = record["question"]
        answer_text: str   = record["answer"]
        
        # Extract entities from both question and answer to build rich links
        q_diseases, q_drugs, q_symptoms = extract_entities(question_text)
        a_diseases, a_drugs, a_symptoms = extract_entities(answer_text)
        
        # Create primary question vertex
        conn.upsertVertex("Question", qid, {"text": question_text, "answer": answer_text})
        
        # Create entity vertices and their corresponding mentions edges
        for disease in set(q_diseases + a_diseases):
            disease_id = disease.lower().replace(" ", "_")
            conn.upsertVertex("Disease", disease_id, {"name": disease})
            conn.upsertEdge("Question", qid, "mentions", "Disease", disease_id)
        
        for drug in set(q_drugs + a_drugs):
            drug_id = drug.lower().replace(" ", "_")
            conn.upsertVertex("Drug", drug_id, {"name": drug})
            conn.upsertEdge("Question", qid, "mentions_drug", "Drug", drug_id)
        
        for symptom in set(q_symptoms + a_symptoms):
            symptom_id = symptom.lower().replace(" ", "_")
            conn.upsertVertex("Symptom", symptom_id, {"name": symptom})
            conn.upsertEdge("Question", qid, "mentions_symptom", "Symptom", symptom_id)
    
    print("✅ Ingestion complete!")
    return conn

if __name__ == "__main__":
    # Integration test with PubMedQA data
    from data.loader import load_pubmedqa
    records = load_pubmedqa(100)
    ingest_to_tigergraph(records)
