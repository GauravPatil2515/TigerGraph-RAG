# graph/schema.py
import pyTigerGraph as tg
from config import TG_HOST, TG_GRAPHNAME, TG_APIKEY

def get_connection():
    conn = tg.TigerGraphConnection(
        host=TG_HOST,
        graphname=TG_GRAPHNAME,
        apiToken=TG_APIKEY          # Savanna uses apiToken, not username/password
    )
    print(f"Connected to: {conn.graphname}")
    return conn
