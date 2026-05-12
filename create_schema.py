"""Create GraphRAG schema in TigerGraph Cloud using pyTigerGraph.

This is the PRIMARY schema setup script for the production pipeline.
Schema: Document + Entity vertices, mentions edge.
Run this ONCE before first ingestion.

NOTE: graph/loader.py and export_csv.py define alternative/demo schemas
      and are NOT used in the production pipeline. See ARCHITECTURE.md.
"""
import pyTigerGraph as tg
import os
from dotenv import load_dotenv

load_dotenv()

host   = os.getenv("TIGERGRAPH_HOST")
secret = os.getenv("TIGERGRAPH_SECRET")
graph  = os.getenv("TIGERGRAPH_GRAPHNAME", "MedGraph")  # fixed: was 'MyGraph'

print(f"Connecting to: {host} | Graph: {graph}")

conn = tg.TigerGraphConnection(
    host=host,
    graphname=graph,
    gsqlSecret=secret,
    tgCloud=True
)
token = conn.getToken(secret)
if isinstance(token, tuple):
    token = token[0]

conn.apiToken = token
conn.authHeader = {'Authorization': 'Bearer ' + token}

print(f"Token: {token[:20]}...")

# Step 1: Create Graph
print(f"\n--- Creating graph '{graph}' ---")
try:
    res = conn.gsql(f"CREATE GRAPH {graph} ()")
    print(res)
except Exception as e:
    print(f"Graph creation info: {e}")

# Step 2: Create Schema
# NOTE: 'related_to' edge is defined for future use (entity-entity relationships).
# Currently only 'mentions' (Document -> Entity) is populated by ingest/tigergraph_ingest.py
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
# Note: related_to (Entity->Entity) removed — not populated in current pipeline.
# Re-add when entity co-occurrence logic is implemented.

print("\n--- Creating vertex/edge types ---")
try:
    res = conn.gsql(schema_gsql)
    print(res)
except Exception as e:
    print(f"Schema creation info: {e}")

# Step 3: Verify
print("\n--- Verifying schema ---")
try:
    print("Vertex types:", conn.getVertexTypes())
    print("Edge types:",   conn.getEdgeTypes())
except Exception as e:
    print(f"Verification error: {e}")
