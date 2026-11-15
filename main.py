import hashlib, json
from mtplatform.tenancy import TenantContext
from mtplatform.retrieval import VectorStore
from mtplatform.ingestion import IngestionService
from mtplatform.llm import ModelClient, Router
from mtplatform.tools import ToolExecutor, Tool
from mtplatform.budget import TenantBudget
from mtplatform.agent import Agent
from mtplatform.ollama_llm import ollama_llm

class FakeEmbedder:
    """Deterministic bag-of-hashes embedding. Replace with a real model."""
    DIM = 64
    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.DIM
            for w in t.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1
            out.append(v)
        return out

def fake_llm(messages, tools):
    # Replace with a real provider SDK call. This one just echoes context.
    last = messages[-1]
    if last["role"] == "tool":
        return {"content": f"Based on the tool result: {last['content']}", "tokens": 50}
    return {"content": "Answer from context: " + messages[1]["content"][:120], "tokens": 80}

def lookup_order(order_id: str) -> dict:
    return {"order_id": order_id, "status": "shipped"}

if __name__ == "__main__":
    embedder = FakeEmbedder()
    store = VectorStore()
    ingest = IngestionService(embedder, store)

    acme = TenantContext("acme", frozenset({"support"}))
    globex = TenantContext("globex", frozenset({"support"}))

    from mtplatform.tenancy import set_tenant, _current
    t = set_tenant(acme)
    ingest.ingest("acme", "refund-policy", "Acme refunds are allowed within 30 days of purchase.")
    _current.reset(t)
    t = set_tenant(globex)
    ingest.ingest("globex", "refund-policy", "Globex never refunds any purchase.")
    _current.reset(t)

    tools = ToolExecutor()
    tools.register(Tool("lookup_order", lookup_order,
                        {"type": "object", "properties": {"order_id": {"type": "string"}}},
                        timeout_s=5, retries=2))

    agent = Agent(
        name="support", system_prompt="You are a support agent. Use context only.",
       router=Router([ModelClient("ollama", ollama_llm, timeout_s=120, retries=1)]),
        store=store, embedder=embedder, tools=tools,
        tool_names=["lookup_order"], budget=TenantBudget(max_tokens=10_000),
    )

    result = agent.run(acme, "What is the refund policy?")
    print(json.dumps(result, indent=2))

    # Isolation check: Globex's agent must not see Acme's policy.
    result = agent.run(globex, "What is the refund policy?")
    assert "30 days" not in result["answer"], "tenant leak!"
    print("Isolation check passed.")