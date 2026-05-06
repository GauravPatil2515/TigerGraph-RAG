import os, sys
sys.path.insert(0, '.')

print("=" * 60)
print("DATABASE CONNECTION DIAGNOSTIC")
print("=" * 60)

# ── 1. Environment Variables ─────────────────────────────────
print("\n📁 1. ENVIRONMENT VARIABLES")
from dotenv import load_dotenv
load_dotenv()

vars_to_check = [
    "GROQ_API_KEY", "TIGERGRAPH_HOST",
    "TIGERGRAPH_GRAPHNAME", "TIGERGRAPH_USERNAME",
    "TIGERGRAPH_PASSWORD", "TIGERGRAPH_SECRET", "CHROMA_PATH",
]
for var in vars_to_check:
    val = os.getenv(var)
    if val:
        print(f"  ✅ {var}: {val[:8]}...")
    else:
        print(f"  ❌ {var}: NOT SET")

# ── 2. ChromaDB ──────────────────────────────────────────────
print("\n🗄️  2. CHROMADB (Basic RAG Vector Database)")
try:
    import chromadb
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collections = client.list_collections()
    print(f"  ✅ Connected at: {CHROMA_PATH}")
    print(f"  ✅ Collections: {len(collections)}")
    for col in collections:
        c = client.get_collection(col.name)
        count = c.count()
        status = "✅" if count >= 100 else "⚠️  LOW"
        print(f"     {status} '{col.name}': {count} documents")
        if count == 0:
            print(f"     🔧 FIX: python ingest/chroma_ingest.py")
except Exception as e:
    print(f"  ❌ ChromaDB ERROR: {e}")
    print(f"     🔧 FIX: python ingest/chroma_ingest.py")

# ── 3. TigerGraph ────────────────────────────────────────────
print("\n🐯 3. TIGERGRAPH (GraphRAG Graph Database)")
try:
    import pyTigerGraph as tg
    TG_HOST = os.getenv("TIGERGRAPH_HOST", "")
    TG_GRAPH = os.getenv("TIGERGRAPH_GRAPHNAME", "MyGraph")
    TG_SECRET = os.getenv("TIGERGRAPH_SECRET", "")
    
    if not TG_HOST:
        print("  ❌ TIGERGRAPH_HOST not set in .env")
    elif not TG_SECRET or "paste_your_secret_here" in TG_SECRET:
        print("  ❌ TIGERGRAPH_SECRET is missing or placeholder")
    else:
        try:
            print(f"  Connecting to: {TG_HOST}...")
            conn = tg.TigerGraphConnection(host=TG_HOST, graphname=TG_GRAPH, gsqlSecret=TG_SECRET, tgCloud=True)
            token = conn.getToken(TG_SECRET)
            if isinstance(token, tuple): token = token[0]
            conn.apiToken = token
            print(f"  ✅ Connected! Token: {str(token)[:15]}...")
            print(f"  ✅ Graph Name: {conn.graphname}")
        except Exception as e:
            if "500" in str(e) or "workspace" in str(e).lower():
                print(f"  ❌ Workspace Error: {e}")
                print(f"     🔧 FIX: Go to tgcloud.io → Resume your workspace")
            else:
                print(f"  ❌ Connection Error: {e}")
except Exception as e:
    print(f"  ❌ TigerGraph check failed: {e}")


# ── 4. Groq API ──────────────────────────────────────────────
print("\n🤖 4. GROQ API (LLM for all pipelines)")
try:
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("  ❌ GROQ_API_KEY not set")
    else:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Reply with just: OK"}],
            max_tokens=5
        )
        print(f"  ✅ Groq API connected — model: llama-3.3-70b-versatile")
        print(f"  ✅ Test response: {response.choices[0].message.content}")
except Exception as e:
    print(f"  ❌ Groq ERROR: {e}")
    print(f"     🔧 FIX: Check GROQ_API_KEY at console.groq.com")

# ── 5. Benchmark Results ─────────────────────────────────────
print("\n📊 5. BENCHMARK RESULTS")
try:
    import glob, pandas as pd
    csv_files = sorted(glob.glob("results/*.csv"))
    if csv_files:
        df = pd.read_csv(csv_files[-1])
        print(f"  ✅ Latest: {csv_files[-1]}")
        print(f"  ✅ Rows: {len(df)} | Pipelines: {df['pipeline_name'].unique().tolist()}")
        required = ['pipeline_name','total_tokens','latency_ms','cost_usd','bert_f1_raw']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  ⚠️  Missing columns: {missing}")
        else:
            print(f"  ✅ All required columns present")
            # Show summary
            summary = df.groupby("pipeline_name")[["total_tokens","latency_ms","cost_usd"]].mean().round(2)
            print(f"\n  📈 PERFORMANCE SUMMARY:")
            print(summary.to_string())
    else:
        print("  ❌ No results CSV found")
        print("     🔧 FIX: python main.py --mode quick")
except Exception as e:
    print(f"  ❌ Results error: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE — paste results above to get fix instructions")
print("=" * 60)
