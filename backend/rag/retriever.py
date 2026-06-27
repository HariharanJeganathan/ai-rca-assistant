"""
retriever.py — ChromaDB RAG
Updated: Use ChromaDB's built-in DefaultEmbeddingFunction
which uses onnxruntime (lightweight, 22MB) instead of torch (2GB).
This works within Render's 512MB free tier limit.
"""

import logging
import os
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class IncidentRetriever:

    def __init__(self, persist_path: str = "./chroma_data"):
        self.persist_path = persist_path
        self._client = None
        self._collection = None
        self._embed_fn = None
        self._is_initialized = False
        logger.info(f"[Retriever] Init. Path: {persist_path}")

    def _initialize(self):
        if self._is_initialized:
            return
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            logger.info("[Retriever] Connecting to ChromaDB...")
            self._client = chromadb.PersistentClient(
                path=self.persist_path,
                settings=ChromaSettings(anonymized_telemetry=False)
            )

            # DefaultEmbeddingFunction uses onnxruntime — no torch needed!
            # Downloads a 22MB ONNX model on first use only
            self._embed_fn = DefaultEmbeddingFunction()

            self._collection = self._client.get_or_create_collection(
                name="incidents",
                embedding_function=self._embed_fn,
                metadata={"hnsw:space": "cosine"}
            )

            logger.info(f"[Retriever] Connected. KB size: {self._collection.count()}")
            self._is_initialized = True

        except Exception as e:
            logger.error(f"[Retriever] Init failed: {e}")
            raise

    async def ingest(self, incident_id, title, description, severity,
                     affected_systems, root_cause=None, resolution=None, metadata=None):
        try:
            self._initialize()
            full_text = self._build_text(title, description, root_cause, resolution, affected_systems)
            doc_metadata = {
                "incident_id": incident_id,
                "title": title,
                "severity": severity,
                "affected_systems": ",".join(affected_systems) if affected_systems else "",
                "ingested_at": datetime.utcnow().isoformat(),
            }
            if metadata:
                doc_metadata.update(metadata)

            self._collection.add(
                ids=[incident_id],
                documents=[full_text],
                metadatas=[doc_metadata]
            )
            logger.info(f"[Retriever] Ingested {incident_id}. Total: {self._collection.count()}")
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                return False
            logger.error(f"[Retriever] Ingest error: {e}")
            raise

    async def search(self, query, top_k=3, severity_filter=None, min_relevance=0.3):
        try:
            self._initialize()
            count = self._collection.count()
            if count == 0:
                return []

            where_filter = None
            if severity_filter:
                where_filter = {"severity": {"$eq": severity_filter}}

            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, count),
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            similar = []
            if not results or not results.get("ids") or not results["ids"][0]:
                return []

            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                similarity = 1 - (distance / 2)
                if similarity < min_relevance:
                    continue
                metadata = results["metadatas"][0][i]
                document = results["documents"][0][i]
                similar.append({
                    "incident_id": doc_id,
                    "title": metadata.get("title", "Unknown"),
                    "severity": metadata.get("severity", "Unknown"),
                    "affected_systems": metadata.get("affected_systems", "").split(","),
                    "summary": document[:500],
                    "similarity_score": round(similarity, 3),
                    "metadata": metadata
                })
            return similar
        except Exception as e:
            logger.error(f"[Retriever] Search error: {e}")
            return []

    async def get_stats(self):
        try:
            self._initialize()
            return {"total_incidents": self._collection.count(), "status": "connected"}
        except Exception as e:
            return {"total_incidents": 0, "status": "error", "error": str(e)}

    async def delete(self, incident_id):
        try:
            self._initialize()
            self._collection.delete(ids=[incident_id])
            return True
        except Exception as e:
            logger.error(f"[Retriever] Delete error: {e}")
            return False

    def _build_text(self, title, description, root_cause, resolution, affected_systems):
        parts = [f"INCIDENT TITLE: {title}", f"DESCRIPTION: {description}"]
        if affected_systems:
            parts.append(f"AFFECTED SYSTEMS: {', '.join(affected_systems)}")
        if root_cause:
            parts.append(f"ROOT CAUSE: {root_cause}")
        if resolution:
            parts.append(f"RESOLUTION: {resolution}")
        return "\n\n".join(parts)

    async def seed_sample_incidents(self):
        samples = [
            {"incident_id": "SAMPLE-001", "title": "Database connection pool exhausted",
             "description": "Production DB connection pool hit max capacity. All queries timing out.",
             "severity": "P1", "affected_systems": ["api-service", "database"],
             "root_cause": "Slow query in v1.2.3 held connections open too long.",
             "resolution": "Rolled back deployment. Increased pool size."},
            {"incident_id": "SAMPLE-002", "title": "Payment service 502 errors after deployment",
             "description": "Payment service returning 502 Bad Gateway after v2.1.0 deploy.",
             "severity": "P1", "affected_systems": ["payment-service", "checkout-api"],
             "root_cause": "Missing PAYMENT_API_KEY env variable in production.",
             "resolution": "Set missing env var. Added startup validation."},
            {"incident_id": "SAMPLE-003", "title": "Memory leak causing OOMKilled every 6 hours",
             "description": "API pods killed by Kubernetes every 6-8 hours due to OOM.",
             "severity": "P2", "affected_systems": ["api-service", "kubernetes"],
             "root_cause": "Background job cached responses without TTL eviction.",
             "resolution": "Added 30min TTL cache eviction. Increased memory limit."},
            {"incident_id": "SAMPLE-004", "title": "Auth service latency spike",
             "description": "Auth service p99 latency spiked to 8 seconds. Users getting logged out.",
             "severity": "P2", "affected_systems": ["auth-service", "redis"],
             "root_cause": "Redis hit memory limit, evicting active sessions.",
             "resolution": "Increased Redis memory to 8GB. Added LRU eviction policy."},
            {"incident_id": "SAMPLE-005", "title": "Firewall change caused connectivity outage",
             "description": "Multiple services lost connectivity after firewall orchestrator swap.",
             "severity": "P3", "affected_systems": ["Citrix", "Firewall", "ROCS"],
             "root_cause": "CHG caused routing table corruption during LCM hardware swap.",
             "resolution": "Auto-resolved after 30 mins. Post-change validation added."},
        ]
        seeded = 0
        for s in samples:
            if await self.ingest(**s):
                seeded += 1
        logger.info(f"[Retriever] Seeded {seeded} samples")
        return seeded


from functools import lru_cache
from config import settings

@lru_cache(maxsize=1)
def get_retriever() -> IncidentRetriever:
    return IncidentRetriever(persist_path=settings.CHROMA_PERSIST_PATH)
