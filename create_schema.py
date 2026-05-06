"""Create GraphRAG schema in TigerGraph Cloud using pyTigerGraph with explicit token handling."""
import pyTigerGraph as tg
import os
from dotenv import load_dotenv

load_dotenv()

host   = os.getenv("TIGERGRAPH_HOST")
secret = os.getenv("TIGERGRAPH_SECRET")
graph  = os.getenv("TIGERGRAPH_GRAPHNAME", "MyGraph")

print(f"Connecting to: {host}")

# Connect
conn = tg.TigerGraphConnection(
    host=host,
    graphname=graph,
    gsqlSecret=secret,
    tgCloud=True
)
token = conn.getToken(secret)
if isinstance(token, tuple):
    token = token[0]

# Manually set tokens to ensure all methods work
conn.apiToken = token
conn.authHeader = {'Authorization': 'Bearer ' + token}

print(f"Token: {token[:20]}...")

# Step 1: Create Graph
print(f"\n--- Creating graph '{graph}' ---")
try:
    res = conn.gsql(f"CREATE GRAPH {graph} ()")
    print(res)
except Exception as e:
    print(f"Error creating graph: {e}")

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
CREATE DIRECTED EDGE related_to (FROM Entity, TO Entity, weight FLOAT)
"""

print("\n--- Creating vertex/edge types ---")
try:
    # We might need to call this in a specific way or wait for graph to be created
    res = conn.gsql(schema_gsql)
    print(res)
except Exception as e:
    print(f"Error creating schema: {e}")

# Step 3: Verify
print("\n--- Verifying schema ---")
try:
    print("Vertex types:", conn.getVertexTypes())
    print("Edge types:", conn.getEdgeTypes())
except Exception as e:
    print(f"Verification error: {e}")
