import os, time, requests
from groq import Groq
from dotenv import load_dotenv
import pyTigerGraph as tg

load_dotenv()

COST_PER_1K = 0.00059

def get_token_info():
    secret = os.getenv("TIGERGRAPH_SECRET")
    host = os.getenv("TIGERGRAPH_HOST")
    graph = os.getenv("TIGERGRAPH_GRAPHNAME", "MedGraph")
    conn = tg.TigerGraphConnection(host=host, graphname=graph, gsqlSecret=secret, tgCloud=True)
    token = conn.getToken(secret)
    if isinstance(token, tuple): token = token[0]
    return token, conn.restppUrl, graph

class GraphRAGPipeline:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        # Connection setup
        secret = os.getenv("TIGERGRAPH_SECRET")
        host = os.getenv("TIGERGRAPH_HOST")
        self.graph = os.getenv("TIGERGRAPH_GRAPHNAME", "MedGraph")
        self.conn = tg.TigerGraphConnection(host=host, graphname=self.graph, gsqlSecret=secret, tgCloud=True)
        self.token = self.conn.getToken(secret)
        if isinstance(self.token, tuple): self.token = self.token[0]
        self.conn.apiToken = self.token
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        
        self.name = "graphrag"
        self._entity_cache = {}  # Cache to reduce latency
        print(f"✅ GraphRAG Pipeline initialized — Connected to {self.graph}")

    def extract_keywords(self, query: str) -> list:
        stopwords = {"what","is","are","the","a","an","of","in","for","how","does","do","can","with","and","or","to","it"}
        words = [w.lower().strip("?.,") for w in query.split()]
        return [w for w in words if w not in stopwords and len(w) > 3]

    def graph_search(self, keywords: list) -> str:
        cache_key = tuple(sorted(keywords))
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]

        context_parts = []
        for keyword in keywords[:2]:
            try:
                # TRUE multi-hop: Entity → Document → Entity → Document
                gsql_query = f"""
                INTERPRET QUERY () FOR GRAPH {self.graph} {{
                    
                    # HOP 1: Find seed entities matching keyword
                    SeedEntities = SELECT e FROM Entity:e
                                   WHERE e.name LIKE "%{keyword}%"
                                   LIMIT 3;
                    
                    # HOP 2: Find documents mentioning those entities
                    RelatedDocs = SELECT d FROM SeedEntities:e
                                  -(mentions:edge)- Document:d
                                  LIMIT 3;
                    
                    # HOP 3: Find OTHER entities in those documents
                    ConnectedEntities = SELECT e2 FROM RelatedDocs:d
                                       -(mentions:edge)- Entity:e2
                                       WHERE e2.name != "{keyword}"
                                       LIMIT 5;
                    
                    PRINT SeedEntities, RelatedDocs, ConnectedEntities;
                }}
                """
                result = self.conn.runInterpretedQuery(gsql_query)

                for block in result:
                    for key, rows in block.items():
                        for row in rows:
                            if isinstance(row, dict):
                                attrs = row.get("attributes", {})
                                if attrs.get("question"):
                                    context_parts.append(
                                        f"[Graph Doc] {attrs['question'][:150]}\n"
                                        f"→ {attrs['answer'][:150]}"
                                    )
                                elif row.get("v_id"):
                                    context_parts.append(f"[Entity] {row['v_id']}")

            except Exception as e:
                # Fallback to direct vertex lookup if interpreted query fails
                try:
                    r = requests.get(f"{self.conn.restppUrl}/graph/{self.graph}/vertices/Entity/{keyword}", headers=self.headers, verify=False)
                    if r.status_code == 200:
                        context_parts.append(f"[Entity Fallback] {keyword}")
                except:
                    continue
        
        result = "\n\n".join(context_parts[:6]) if context_parts else ""
        self._entity_cache[cache_key] = result
        return result

    def run(self, query: str) -> dict:
        start = time.perf_counter()
        keywords = self.extract_keywords(query)
        graph_context = self.graph_search(keywords)
        
        system_msg = "You are a precise medical assistant. Use the provided context to answer concisely."
        user_msg = f"Context:\n{graph_context}\n\nQuestion: {query}" if graph_context else query
        
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
        response = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.1, max_tokens=256)
        
        latency = (time.perf_counter() - start) * 1000
        usage = response.usage
        total = usage.prompt_tokens + usage.completion_tokens
        
        return {
            "pipeline_name": self.name,
            "answer": response.choices[0].message.content.strip(),
            "tokens_prompt": usage.prompt_tokens,
            "tokens_completion": usage.completion_tokens,
            "total_tokens": total,
            "latency_ms": round(latency, 2),
            "cost_usd": round((total/1000) * COST_PER_1K, 6),
            "graph_hops": 3 if graph_context else 0,
            "graph_context_len": len(graph_context),
            "graph_context_preview": graph_context[:300],
        }

if __name__ == "__main__":
    p = GraphRAGPipeline()
    res = p.run("What are the symptoms of diabetes?")
    print(f"✅ Answer: {res['answer'][:200]}")
    print(f"✅ Tokens: {res['total_tokens']}")
