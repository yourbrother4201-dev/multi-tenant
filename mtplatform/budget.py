import threading

class BudgetExceeded(Exception):
    pass

class TenantBudget:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.used = 0
        self._lock = threading.Lock()

    def charge(self, tokens: int):
        with self._lock:
            if self.used + tokens > self.max_tokens:
                raise BudgetExceeded(f"used {self.used}/{self.max_tokens}")
            self.used += tokens