import time
from mtplatform.tenancy import TenantContext, set_tenant, _current
from mtplatform.retrieval import VectorStore, Chunk
from mtplatform.llm import ModelClient, Router
from mtplatform.tools import ToolExecutor, Tool
from mtplatform.budget import TenantBudget
from mtplatform.agent import Agent

class FakeEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

def slow_llm(messages, tools=None):
    time.sleep(5)  # longer than the timeout below
    return {"content": "late answer", "tokens": 1}

def test_llm_timeout_returns_partial():
    store = VectorStore()
    store.upsert([Chunk("acme", "d1", "policy text", [1.0, 0.0], frozenset())])
    agent = Agent(
        name="t", system_prompt="test",
        router=Router([ModelClient("slow", slow_llm, timeout_s=0.5, retries=0)]),
        store=store, embedder=FakeEmbedder(), tools=ToolExecutor(),
        tool_names=[], budget=TenantBudget(max_tokens=10_000),
    )
    result = agent.run(TenantContext("acme"), "question?")
    assert result["status"] == "partial"

def test_tool_timeout_returns_error():
    tools = ToolExecutor()
    def hang(**kwargs):
        time.sleep(5)
        return {}
    tools.register(Tool("hang", hang, {"type": "object"}, timeout_s=0.2, retries=0))
    out = tools.run("acme", "hang", {})
    assert out.get("error") == "timeout"