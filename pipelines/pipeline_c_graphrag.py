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
        self.token, self.restpp_url, self.graph = get_token_info()
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
        for kw in keywords[:3]:
            try:
                # Search for Entity vertex (simplified)
                # In a real system, we'd use a search endpoint or index.
                # Here we'll try to get the vertex directly by ID (keyword)
                r = requests.get(f"{self.restpp_url}/graph/{self.graph}/vertices/Entity/{kw}", headers=self.headers, verify=False)
                if r.status_code == 200:
                    ent_data = r.json()
                    if not ent_data.get("error"):
                        context_parts.append(f"Entity: {kw}")
                        # Hop 2: Find Document vertices (mentions)
                        r2 = requests.get(f"{self.restpp_url}/graph/{self.graph}/edges/Document/mentions/Entity/{kw}", headers=self.headers, verify=False)
                        if r2.status_code == 200:
                            edges = r2.json().get("results", [])
                            for edge in edges[:2]:
                                doc_id = edge.get("from_id")
                                r3 = requests.get(f"{self.restpp_url}/graph/{self.graph}/vertices/Document/{doc_id}", headers=self.headers, verify=False)
                                if r3.status_code == 200:
                                    doc = r3.json().get("results", [{}])[0]
                                    attrs = doc.get("attributes", {})
                                    q = attrs.get("question", "")[:200]
                                    a = attrs.get("answer", "")[:200]
                                    context_parts.append(f"Q: {q}\nA: {a}")
            except:
                continue
        
        result = "\n\n".join(context_parts[:5]) if context_parts else ""
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
            "graph_context_len": len(graph_context),
            "graph_hops": 2 if graph_context else 0,
        }

if __name__ == "__main__":
    p = GraphRAGPipeline()
    res = p.run("What are the symptoms of diabetes?")
    print(f"✅ Answer: {res['answer'][:200]}")
    print(f"✅ Tokens: {res['total_tokens']}")
