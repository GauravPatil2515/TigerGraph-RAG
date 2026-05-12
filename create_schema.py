"""Create GraphRAG schema in TigerGraph Cloud using pyTigerGraph.

This is the ACTIVE schema setup script for this project.
Run this ONCE before ingesting data.

Active schema:
    Vertices: Document (question, answer, context, source)
              Entity  (entity_type)
    Edges:    mentions (Document → Entity)  [directed]

Deprecated / not used at runtime:
    graph/loader.py   — alternative spaCy-based schema (orphaned experiment)
    export_csv.py     — hardcoded demo CSV export (not production data)
"""
import pyTigerGraph as tg
import os
from dotenv import load_dotenv

load_dotenv()

host   = os.getenv("TIGERGRAPH_HOST")
secret = os.getenv("TIGERGRAPH_SECRET")
graph  = os.getenv("TIGERGRAPH_GRAPHNAME", "MedGraph")  # FIX: was "MyGraph"

print(f"Connecting to: {host}")

conn = tg.TigerGraphConnection(
    host=host,
    graphname=graph,
    gsqlSecret=secret,
    tgCloud=True
)
token = conn.getToken(secret)
if isinstance(token, tuple):
    token = token[0]

conn.apiToken    = token
conn.authHeader  = {'Authorization': 'Bearer ' + token}

print(f"Token: {token[:20]}...")

# Step 1: Create Graph
print(f"\n--- Creating graph '{graph}' ---")
try:
    res = conn.gsql(f"CREATE GRAPH {graph} ()")
    print(res)
except Exception as e:
    print(f"Graph note: {e}")

# Step 2: Create Schema
schema_gsql = f"""
USE GRAPH {graph}

CREATE VERTEX Document (
    PRIMARY_ID doc_id STRING,
    question   STRING,
    answer     STRING,
    context    STRING,
    source     STRING
) WITH primary_id_as_attribute="true"

CREATE VERTEX Entity (
    PRIMARY_ID name STRING,
    entity_type STRING
) WITH primary_id_as_attribute="true"

CREATE DIRECTED EDGE mentions (FROM Document, TO Entity)
"""
# NOTE: related_to (Entity → Entity) edge removed — was never populated
# Add it back when implementing entity co-occurrence scoring in future.

print("\n--- Creating vertex/edge types ---")
try:
    res = conn.gsql(schema_gsql)
    print(res)
except Exception as e:
    print(f"Schema note: {e}")

# Step 3: Verify
print("\n--- Verifying schema ---")
try:
    print("Vertex types:", conn.getVertexTypes())
    print("Edge types:",   conn.getEdgeTypes())
except Exception as e:
    print(f"Verification error: {e}")
