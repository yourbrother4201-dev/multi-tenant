from dataclasses import dataclass
from contextvars import ContextVar

@dataclass(frozen=True)
class TenantContext:
    tenant_id: str
    groups: frozenset[str] = frozenset()

_current: ContextVar[TenantContext | None] = ContextVar("tenant", default=None)

def set_tenant(ctx: TenantContext):
    return _current.set(ctx)

def get_tenant() -> TenantContext:
    ctx = _current.get()
    if ctx is None:
        raise PermissionError("No tenant context bound to this request")
    return ctx