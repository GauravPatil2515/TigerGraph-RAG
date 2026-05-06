"""
Module: export_csv.py
Description: Utility script to generate structured CSV data for TigerGraph 
             bulk ingestion. Exports nodes (Diseases, Drugs, Symptoms) and 
             edges (Treats, Causes) to the csv_export/ directory.

Author: Gaurav Patil
Project: GraphRAG Inference Hackathon — TigerGraph 2026
"""

from data.loader import load_pubmedqa
import pandas as pd, os

os.makedirs("csv_export", exist_ok=True)
# Load a subset of records to help define the graph schema context
records = load_pubmedqa(500)

# Diseases nodes
pd.DataFrame([
    {"name": "diabetes",      "description": "metabolic disease with high blood sugar"},
    {"name": "hypertension",  "description": "high blood pressure condition"},
    {"name": "asthma",        "description": "chronic respiratory condition"},
    {"name": "anemia",        "description": "low red blood cell count"},
    {"name": "depression",    "description": "mental health disorder"},
    {"name": "kidney disease","description": "chronic kidney function loss"},
    {"name": "arthritis",     "description": "joint inflammation disease"},
    {"name": "obesity",       "description": "excess body weight condition"},
]).to_csv("csv_export/diseases.csv", index=False)

# Drugs nodes
pd.DataFrame([
    {"name": "metformin",       "treats": "diabetes",     "side_effect": "nausea"},
    {"name": "lisinopril",      "treats": "hypertension", "side_effect": "dry cough"},
    {"name": "albuterol",       "treats": "asthma",       "side_effect": "tremors"},
    {"name": "iron supplement", "treats": "anemia",       "side_effect": "constipation"},
    {"name": "sertraline",      "treats": "depression",   "side_effect": "insomnia"},
    {"name": "ibuprofen",       "treats": "arthritis",    "side_effect": "stomach upset"},
    {"name": "atorvastatin",    "treats": "obesity",      "side_effect": "muscle pain"},
    {"name": "amlodipine",      "treats": "hypertension", "side_effect": "swelling"},
]).to_csv("csv_export/drugs.csv", index=False)

# Symptoms nodes
pd.DataFrame([
    {"name": "fatigue",           "caused_by": "diabetes"},
    {"name": "frequent urination","caused_by": "diabetes"},
    {"name": "headache",          "caused_by": "hypertension"},
    {"name": "wheezing",          "caused_by": "asthma"},
    {"name": "weakness",          "caused_by": "anemia"},
    {"name": "sadness",           "caused_by": "depression"},
    {"name": "joint pain",        "caused_by": "arthritis"},
    {"name": "swollen ankles",    "caused_by": "kidney disease"},
]).to_csv("csv_export/symptoms.csv", index=False)

# Edges — Drug treats Disease
pd.DataFrame([
    {"drug": "metformin",       "disease": "diabetes"},
    {"drug": "lisinopril",      "disease": "hypertension"},
    {"drug": "amlodipine",      "disease": "hypertension"},
    {"drug": "albuterol",       "disease": "asthma"},
    {"drug": "iron supplement", "disease": "anemia"},
    {"drug": "sertraline",      "disease": "depression"},
    {"drug": "ibuprofen",       "disease": "arthritis"},
]).to_csv("csv_export/treats_edges.csv", index=False)

# Edges — Disease causes Symptom
pd.DataFrame([
    {"disease": "diabetes",      "symptom": "fatigue"},
    {"disease": "diabetes",      "symptom": "frequent urination"},
    {"disease": "hypertension",  "symptom": "headache"},
    {"disease": "asthma",        "symptom": "wheezing"},
    {"disease": "anemia",        "symptom": "weakness"},
    {"disease": "depression",    "symptom": "sadness"},
    {"disease": "arthritis",     "symptom": "joint pain"},
    {"disease": "kidney disease","symptom": "swollen ankles"},
]).to_csv("csv_export/causes_edges.csv", index=False)

print("✅ CSV files ready in csv_export/:")
print(os.listdir("csv_export"))
