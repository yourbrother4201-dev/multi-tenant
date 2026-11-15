from mtplatform.tenancy import TenantContext, set_tenant, _current
from mtplatform.retrieval import VectorStore, Chunk

def test_cross_tenant_search_returns_nothing():
    store = VectorStore()
    store.upsert([Chunk("acme", "d1", "secret", [1.0, 0.0], frozenset())])
    token = set_tenant(TenantContext("globex"))
    try:
        assert store.search([1.0, 0.0]) == []
    finally:
        _current.reset(token)

def test_acl_blocks_unauthorized_group():
    store = VectorStore()
    store.upsert([Chunk("acme", "d1", "hr only", [1.0, 0.0], frozenset({"hr"}))])
    token = set_tenant(TenantContext("acme", frozenset({"support"})))
    try:
        assert store.search([1.0, 0.0]) == []
    finally:
        _current.reset(token)