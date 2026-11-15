import time, random
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout

class LLMError(Exception):
    pass

class ModelClient:
    """Wraps one provider call with per-attempt timeout and retries."""

    def __init__(self, name: str, call_fn, timeout_s: float = 20, retries: int = 2):
        self.name = name
        self._call = call_fn
        self.timeout_s = timeout_s
        self.retries = retries
        self._pool = ThreadPoolExecutor(max_workers=8)

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        last_err = None
        for attempt in range(self.retries + 1):
            fut = self._pool.submit(self._call, messages, tools)
            try:
                return fut.result(timeout=self.timeout_s)
            except FTimeout:
                last_err = "timeout"
                fut.cancel()
            except Exception as e:  # provider 5xx, rate limit, etc.
                last_err = str(e)
            time.sleep(min(8, 2 ** attempt) * random.uniform(0.5, 1.0))
        raise LLMError(f"{self.name} failed: {last_err}")


class Router:
    """Tries models in order; the first healthy one wins."""

    def __init__(self, clients: list[ModelClient]):
        self.clients = clients

    def complete(self, messages, tools=None):
        errors = []
        for c in self.clients:
            try:
                return c.complete(messages, tools)
            except LLMError as e:
                errors.append(str(e))
        raise LLMError("all models failed: " + "; ".join(errors))