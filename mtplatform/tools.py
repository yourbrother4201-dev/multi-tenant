import concurrent.futures as cf
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class Tool:
    name: str
    fn: Callable[..., dict]
    schema: dict            # JSON schema for arguments
    timeout_s: float = 10
    retries: int = 0        # only set >0 for read-only tools
    allowed_tenants: set[str] = field(default_factory=set)  # empty = all

class ToolExecutor:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._pool = cf.ThreadPoolExecutor(max_workers=16)

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def run(self, tenant_id: str, name: str, args: dict) -> dict:
        tool = self._tools.get(name)
        if tool is None:
            return {"error": "unknown_tool", "retryable": False}
        if tool.allowed_tenants and tenant_id not in tool.allowed_tenants:
            return {"error": "forbidden", "retryable": False}

        for attempt in range(tool.retries + 1):
            fut = self._pool.submit(tool.fn, **args)
            try:
                return {"result": fut.result(timeout=tool.timeout_s)}
            except cf.TimeoutError:
                fut.cancel()
                err = "timeout"
            except Exception as e:
                err = str(e)
        # Return errors as observations; the model decides what to do next.
        return {"error": err, "retryable": tool.retries > 0}