import hashlib
from .retrieval import Chunk, VectorStore

def chunk_text(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    words = text.split()
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i + size]))
        i += size - overlap
    return out

class IngestionService:
    def __init__(self, embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store
        self._seen: set[tuple[str, str]] = set()  # (tenant, content_hash)

    def ingest(self, tenant_id: str, doc_id: str, text: str,
               acl_groups: frozenset[str] = frozenset()) -> int:
        digest = hashlib.sha256(text.encode()).hexdigest()
        if (tenant_id, digest) in self._seen:
            return 0  # idempotent re-upload
        self._seen.add((tenant_id, digest))

        pieces = chunk_text(text)
        vectors = self.embedder.embed(pieces)
        self.store.upsert([
            Chunk(tenant_id, doc_id, p, v, acl_groups)
            for p, v in zip(pieces, vectors)
        ])
        return len(pieces)