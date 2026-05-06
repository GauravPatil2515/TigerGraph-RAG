# graph/schema.py
import pyTigerGraph as tg
import os
from config import TG_HOST, TG_GRAPHNAME, TG_SECRET

def get_connection():
    """Connect to TigerGraph Cloud (Savanna) instance"""
    host   = TG_HOST
    graph  = TG_GRAPHNAME
    secret = TG_SECRET

    conn = tg.TigerGraphConnection(
        host=host,
        graphname=graph,
        gsqlSecret=secret,
        tgCloud=True          # ← mandatory for .i.tgcloud.io hosts
    )
    conn.getToken(secret)
    return conn
