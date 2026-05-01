# graph/loader.py
import spacy
from tqdm import tqdm
from graph.schema import get_connection

nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str):
    """Extract medical entities using spaCy NER."""
    doc = nlp(text)
    diseases = [ent.text for ent in doc.ents if ent.label_ in ["DISEASE", "CONDITION"]]
    drugs = [ent.text for ent in doc.ents if ent.label_ in ["CHEMICAL", "DRUG"]]
    symptoms = [ent.text for ent in doc.ents if ent.label_ in ["SYMPTOM", "SIGN"]]
    return diseases, drugs, symptoms

def create_schema(conn):
    """Create graph schema for medical knowledge graph."""
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
        
        print("Schema created successfully!")
    except Exception as e:
        print(f"Schema may already exist: {e}")

def ingest_to_tigergraph(records: list, batch_size: 100):
    """Ingest MedQuAD data into TigerGraph as knowledge graph."""
    conn = get_connection()
    
    # Create schema if not exists
    create_schema(conn)
    
    print(f"Ingesting {len(records)} records into TigerGraph...")
    
    for i, record in enumerate(tqdm(records)):
        qid = f"q_{i}"
        question_text = record["question"]
        answer_text = record["answer"]
        
        # Extract entities from both question and answer
        q_diseases, q_drugs, q_symptoms = extract_entities(question_text)
        a_diseases, a_drugs, a_symptoms = extract_entities(answer_text)
        
        # Create question vertex
        conn.upsertVertex("Question", qid, {"text": question_text, "answer": answer_text})
        
        # Create entity vertices and edges
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
    
    print("Ingestion complete!")
    return conn

if __name__ == "__main__":
    from data.loader import load_medquad
    records = load_medquad(100)
    ingest_to_tigergraph(records)
