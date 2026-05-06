"""
Module: schema.py
Description: Graph connection utility for TigerGraph Cloud. 
             Provides a standardized way to initialize the pyTigerGraph 
             connection across the project.

Author: Gaurav Patil
Project: GraphRAG Inference Hackathon — TigerGraph 2026
"""

import pyTigerGraph as tg
import os
from config import TG_HOST, TG_GRAPHNAME, TG_SECRET

def get_connection() -> tg.TigerGraphConnection:
    """
    Establish a connection to the TigerGraph Cloud (Savanna) instance.
    
    Uses environment variables for host, graph name, and secret. 
    Automatically handles token generation for REST++ authentication.

    Returns:
        tg.TigerGraphConnection: Initialized and authenticated connection object.
    """
    host: str   = TG_HOST
    graph: str  = TG_GRAPHNAME
    secret: str = TG_SECRET

    conn = tg.TigerGraphConnection(
        host=host,
        graphname=graph,
        gsqlSecret=secret,
        tgCloud=True          # Required for TigerGraph Cloud .i.tgcloud.io hosts
    )
    conn.getToken(secret)
    return conn
