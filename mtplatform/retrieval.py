import math
from dataclasses import dataclass
from collections import defaultdict
from .tenancy import get_tenant

@dataclass
class Chunk:
    tenant_id: str
    doc_id: str
    text: str
    vector: list[float]
    acl_groups: frozenset[str]

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0

class VectorStore:
    """In-memory stand-in. Partitioned by tenant; every read is filtered."""

    def __init__(self):
        self._by_tenant: dict[str, list[Chunk]] = defaultdict(list)

    def upsert(self, chunks: list[Chunk]):
        for c in chunks:
            self._by_tenant[c.tenant_id].append(c)

    def search(self, query_vec, k=5) -> list[tuple[float, Chunk]]:
        ctx = get_tenant()  # tenant comes from context, never from the caller
        # Partition lookup + ACL filter. Caller cannot widen either.
        visible = [
            c for c in self._by_tenant[ctx.tenant_id]
            if not c.acl_groups or c.acl_groups & ctx.groups
        ]
        scored = [(cosine(query_vec, c.vector), c) for c in visible]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]