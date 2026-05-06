import os, re, sys, time, requests
from tqdm import tqdm
from dotenv import load_dotenv
import pyTigerGraph as tg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.loader import load_pubmedqa

load_dotenv()

def get_token():
    secret = os.getenv("TIGERGRAPH_SECRET")
    host = os.getenv("TIGERGRAPH_HOST")
    graph = os.getenv("TIGERGRAPH_GRAPHNAME", "MedGraph")
    conn = tg.TigerGraphConnection(host=host, graphname=graph, gsqlSecret=secret, tgCloud=True)
    token = conn.getToken(secret)
    if isinstance(token, tuple): token = token[0]
    return token, conn.restppUrl

def extract_entities(text: str) -> list:
    medical_terms = ["diabetes", "hypertension", "insulin", "glucose", "cancer", "metformin", "asthma", "obesity", "depression", "arthritis", "kidney", "liver", "heart", "lung", "brain", "blood", "therapy", "treatment", "diagnosis", "symptoms", "medication", "surgery", "clinical", "patient", "disease", "disorder", "cholesterol", "pressure", "immune", "infection", "virus"]
    found = []
    text_lower = text.lower()
    for term in medical_terms:
        if term in text_lower: found.append(term)
    return list(set(found))

def ingest_documents(records: list, limit: int = 500):
    token, restpp_url = get_token()
    graph = os.getenv("TIGERGRAPH_GRAPHNAME", "MedGraph")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    print(f"Ingesting {min(limit, len(records))} documents into {graph}...")
    
    docs_upserted = 0
    entities_upserted = 0
    edges_upserted = 0

    for i, rec in enumerate(tqdm(records[:limit])):
        doc_id = f"pubmed_{i}"
        question = rec.get("question", "")[:500]
        answer = rec.get("answer", "")[:500]
        context = rec.get("context", "")[:300]

        # Upsert Document
        payload = {"vertices": {"Document": {doc_id: {"question": {"value": question}, "answer": {"value": answer}, "context": {"value": context}, "source": {"value": "PubMedQA"}}}}}
        r = requests.post(f"{restpp_url}/graph/{graph}", json=payload, headers=headers, verify=False)
        if r.status_code == 200: docs_upserted += 1

        # Extract and upsert entities
        entities = extract_entities(question + " " + answer)
        for ent in entities:
            # Vertex
            payload = {"vertices": {"Entity": {ent: {"entity_type": {"value": "medical_term"}}}}}
            requests.post(f"{restpp_url}/graph/{graph}", json=payload, headers=headers, verify=False)
            
            # Edge
            payload = {"edges": {"Document": {doc_id: {"mentions": {"Entity": {ent: {}}}}}}}
            requests.post(f"{restpp_url}/graph/{graph}", json=payload, headers=headers, verify=False)
            entities_upserted += 1
            edges_upserted += 1

    print(f"\n✅ Ingestion complete! Docs: {docs_upserted}, Entities: {entities_upserted}, Edges: {edges_upserted}")

if __name__ == "__main__":
    try:
        from datasets import load_dataset
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        records = [{"question": r["question"], "answer": r["long_answer"], "context": " ".join(r["context"]["contexts"])[:500]} for r in ds]
    except:
        records = load_pubmedqa(500)
    ingest_documents(records, limit=100) # Reduced limit for speed in testing
