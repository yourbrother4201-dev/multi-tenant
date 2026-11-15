import json
from .tenancy import TenantContext, set_tenant, get_tenant
from .retrieval import VectorStore
from .llm import Router, LLMError
from .tools import ToolExecutor
from .budget import TenantBudget, BudgetExceeded

class Agent:
    def __init__(self, name: str, system_prompt: str, router: Router,
                 store: VectorStore, embedder, tools: ToolExecutor,
                 tool_names: list[str], budget: TenantBudget,
                 max_steps: int = 8, top_k: int = 5):
        self.name = name
        self.system_prompt = system_prompt
        self.router = router
        self.store = store
        self.embedder = embedder
        self.tools = tools
        self.tool_names = tool_names
        self.budget = budget
        self.max_steps = max_steps
        self.top_k = top_k

    def run(self, ctx: TenantContext, question: str) -> dict:
        token = set_tenant(ctx)
        try:
            return self._loop(ctx, question)
        finally:
            from .tenancy import _current
            _current.reset(token)

    def _loop(self, ctx: TenantContext, question: str) -> dict:
        # RAG: retrieve tenant-visible context first.
        hits = self.store.search(self.embedder.embed([question])[0], k=self.top_k)
        context = "\n---\n".join(c.text for _, c in hits)

        messages = [
            {"role": "system", "content": self.system_prompt},
            # Retrieved text is data, not instructions.
            {"role": "user", "content":
                f"<context>\n{context}\n</context>\n\nQuestion: {question}"},
        ]
        tool_schemas = [
            {"name": n, "schema": self.tools._tools[n].schema}
            for n in self.tool_names if n in self.tools._tools
        ]
        trace = []

        for step in range(self.max_steps):
            try:
                reply = self.router.complete(messages, tool_schemas)
                self.budget.charge(reply.get("tokens", 0))
            except BudgetExceeded as e:
                return {"status": "budget_exceeded", "answer": None, "trace": trace}
            except LLMError as e:
                return {"status": "partial", "answer": None,
                        "error": str(e), "trace": trace}

            if not reply.get("tool_calls"):
                return {"status": "succeeded", "answer": reply["content"], "trace": trace}

            messages.append({"role": "assistant", "content": reply.get("content", ""),
                             "tool_calls": reply["tool_calls"]})
            for call in reply["tool_calls"]:
                args = call["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                result = self.tools.run(ctx.tenant_id, call["name"], args)
                trace.append({"step": step, "tool": call["name"], "result": result})
                messages.append({"role": "tool", "tool_call_id": call.get("id"),
                                 "content": json.dumps(result)})

        return {"status": "max_steps", "answer": None, "trace": trace}