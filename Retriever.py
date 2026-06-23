"""
retriever.py — ChromaDB RAG (Retrieval-Augmented Generation)
=============================================================
RAG = Retrieval-Augmented Generation

WHAT THIS DOES IN PLAIN ENGLISH:
  Imagine you are a doctor.
  Before diagnosing a new patient, you check your files for
  patients with similar symptoms. That history helps you diagnose better.

  That's EXACTLY what RAG does for our AI:
    1. We store all past incidents in ChromaDB (like a filing cabinet)
    2. When a new incident arrives, we search ChromaDB for similar ones
    3. We give those similar incidents to the AI as extra context
    4. The AI gives a much better RCA because it has seen this before

HOW CHROMADB WORKS:
  Normal search (Google-style): match keywords
    "database error" finds documents with those exact words

  ChromaDB (vector search): match meaning
    "database error" also finds "DB connection refused", "SQL timeout"
    Because their MEANING is similar, even if words differ!

  This is done using "embeddings" — converting text to numbers
  that capture the meaning. Similar meaning = similar numbers = found!

FLOW:
  Text → Embedding Model → [0.2, 0.8, 0.1, ...] (vector of 384 numbers)
  Store vectors in ChromaDB
  New incident → same embedding → find nearest vectors → return similar incidents
"""

import logging
import os
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class IncidentRetriever:
    """
    Handles storing and searching past incidents using ChromaDB.

    Two main operations:
      1. ingest()  — Add an incident to the knowledge base
      2. search()  — Find similar incidents for a new query

    Uses HuggingFace embeddings (free, runs locally) by default.
    """

    def __init__(self, persist_path: str = "./chroma_data"):
        """
        Initialize ChromaDB client and embedding model.

        persist_path = where ChromaDB saves its data on disk
        So the memory survives app restarts!
        """
        self.persist_path = persist_path
        self._client = None         # ChromaDB client (lazy loaded)
        self._collection = None     # The "table" inside ChromaDB
        self._embeddings = None     # Embedding model (lazy loaded)
        self._is_initialized = False

        logger.info(f"[Retriever] Initialized. Data path: {persist_path}")

    def _initialize(self):
        """
        Lazy initialization — only connect to ChromaDB when first needed.
        This avoids slowing down app startup.

        LAZY LOADING PATTERN:
          Don't connect to external services at import time.
          Connect only when the first request needs it.
          Common pattern in production code.
        """
        if self._is_initialized:
            return

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            logger.info("[Retriever] Connecting to ChromaDB...")

            # Create ChromaDB client with persistent storage
            # PersistentClient saves data to disk (survives restarts)
            self._client = chromadb.PersistentClient(
                path=self.persist_path,
                settings=ChromaSettings(anonymized_telemetry=False)
            )

            # Get or create a "collection" (like a table in SQL)
            # Collection name: "incidents" — stores all past incidents
            self._collection = self._client.get_or_create_collection(
                name="incidents",
                metadata={
                    "description": "Past incident reports for RAG search",
                    # cosine similarity = best for text similarity
                    "hnsw:space": "cosine"
                }
            )

            logger.info(
                f"[Retriever] ChromaDB connected. "
                f"Collection 'incidents' has {self._collection.count()} items"
            )

            # Load embedding model
            self._load_embeddings()

            self._is_initialized = True

        except ImportError:
            raise ImportError(
                "chromadb not installed. Run: pip install chromadb"
            )
        except Exception as e:
            logger.error(f"[Retriever] Failed to initialize ChromaDB: {e}")
            raise

    def _load_embeddings(self):
        """
        Load the embedding model.
        Embeddings convert text → numbers (vectors).

        We use HuggingFace's all-MiniLM-L6-v2:
          - FREE (no API key)
          - Runs locally on CPU (no GPU needed)
          - Small and fast (22MB model)
          - 384-dimensional vectors (good quality)
        """
        from config import get_embeddings
        self._embeddings = get_embeddings()
        logger.info("[Retriever] Embedding model loaded")

    def _embed_text(self, text: str) -> List[float]:
        """
        Convert text to a vector (list of numbers).
        ChromaDB uses these numbers to find similar texts.

        Example:
          "database connection failed" → [0.12, 0.87, 0.34, ...]
          "DB connection refused"      → [0.11, 0.85, 0.36, ...]
          (similar meaning = similar numbers!)
        """
        self._initialize()
        embedding = self._embeddings.embed_query(text)
        return embedding

    # ============================================================
    # 1. INGEST — Add incident to knowledge base
    # ============================================================
    async def ingest(
        self,
        incident_id: str,
        title: str,
        description: str,
        severity: str,
        affected_systems: List[str],
        root_cause: Optional[str] = None,
        resolution: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add one incident to the ChromaDB knowledge base.

        What gets stored:
          - The TEXT of the incident (for display)
          - The VECTOR of the incident (for search)
          - METADATA (severity, date, etc.) for filtering

        Args:
            incident_id: Unique ID like "INC-20241201-A3B4"
            title: Short title of the incident
            description: Full description
            severity: P1, P2, P3, or P4
            affected_systems: List of impacted services
            root_cause: What caused it (if known)
            resolution: How it was fixed (if known)

        Returns:
            True if successfully added, False if already exists
        """
        try:
            self._initialize()

            # Build the full text that will be embedded
            # More context = better search results
            full_text = self._build_document_text(
                title=title,
                description=description,
                root_cause=root_cause,
                resolution=resolution,
                affected_systems=affected_systems
            )

            # Convert text to vector
            embedding = self._embed_text(full_text)

            # Build metadata (used for filtering in search)
            doc_metadata = {
                "incident_id": incident_id,
                "title": title,
                "severity": severity,
                "affected_systems": ",".join(affected_systems) if affected_systems else "",
                "has_root_cause": bool(root_cause),
                "has_resolution": bool(resolution),
                "ingested_at": datetime.utcnow().isoformat(),
            }

            # Merge any extra metadata passed in
            if metadata:
                doc_metadata.update(metadata)

            # Add to ChromaDB
            # ChromaDB stores: document text + embedding + metadata + ID
            self._collection.add(
                ids=[incident_id],
                embeddings=[embedding],
                documents=[full_text],
                metadatas=[doc_metadata]
            )

            count = self._collection.count()
            logger.info(
                f"[Retriever] Ingested incident '{incident_id}'. "
                f"Total in KB: {count}"
            )
            return True

        except Exception as e:
            # If the ID already exists, ChromaDB raises an error
            if "already exists" in str(e).lower():
                logger.warning(f"[Retriever] Incident {incident_id} already in KB")
                return False
            logger.error(f"[Retriever] Error ingesting {incident_id}: {e}")
            raise

    # ============================================================
    # 2. SEARCH — Find similar incidents
    # ============================================================
    async def search(
        self,
        query: str,
        top_k: int = 3,
        severity_filter: Optional[str] = None,
        min_relevance: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Search for incidents similar to the query text.

        HOW IT WORKS:
          1. Convert query to a vector using the same embedding model
          2. ChromaDB finds the K nearest vectors (most similar)
          3. Return those incidents as context for the AI

        Args:
            query: The text to search for (e.g. new incident description)
            top_k: How many similar incidents to return (default: 3)
            severity_filter: Only return incidents of this severity
            min_relevance: Minimum similarity score (0-1, higher = more similar)

        Returns:
            List of similar incidents with their metadata
        """
        try:
            self._initialize()

            # Check if knowledge base is empty
            count = self._collection.count()
            if count == 0:
                logger.info("[Retriever] Knowledge base is empty, no similar incidents found")
                return []

            # Convert query to vector
            query_embedding = self._embed_text(query)

            # Build optional filter
            where_filter = None
            if severity_filter:
                where_filter = {"severity": {"$eq": severity_filter}}

            # Search ChromaDB
            # n_results = how many to return
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, count),  # Can't request more than what exists
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            # Process results
            similar_incidents = []

            if not results or not results.get("ids") or not results["ids"][0]:
                return []

            for i, doc_id in enumerate(results["ids"][0]):
                # Distance in ChromaDB with cosine:
                # 0.0 = identical, 2.0 = completely opposite
                # We convert to similarity: 1 - (distance/2)
                distance = results["distances"][0][i]
                similarity = 1 - (distance / 2)

                # Skip results below minimum relevance
                if similarity < min_relevance:
                    logger.info(
                        f"[Retriever] Skipping '{doc_id}' "
                        f"(similarity {similarity:.2f} < {min_relevance})"
                    )
                    continue

                metadata = results["metadatas"][0][i]
                document = results["documents"][0][i]

                similar_incidents.append({
                    "incident_id": doc_id,
                    "title": metadata.get("title", "Unknown"),
                    "severity": metadata.get("severity", "Unknown"),
                    "affected_systems": metadata.get("affected_systems", "").split(","),
                    "summary": document[:500],  # First 500 chars as preview
                    "similarity_score": round(similarity, 3),
                    "metadata": metadata
                })

            logger.info(
                f"[Retriever] Search complete: "
                f"found {len(similar_incidents)} similar incidents "
                f"(out of {count} in KB)"
            )
            return similar_incidents

        except Exception as e:
            logger.error(f"[Retriever] Search error: {e}")
            # Return empty list instead of crashing
            # The RCA can still work without similar incidents
            return []

    # ============================================================
    # 3. GET STATS — How many incidents in the knowledge base?
    # ============================================================
    async def get_stats(self) -> Dict[str, Any]:
        """
        Return statistics about the knowledge base.
        """
        try:
            self._initialize()
            count = self._collection.count()

            return {
                "total_incidents": count,
                "persist_path": self.persist_path,
                "status": "connected"
            }

        except Exception as e:
            logger.error(f"[Retriever] Stats error: {e}")
            return {
                "total_incidents": 0,
                "status": "error",
                "error": str(e)
            }

    # ============================================================
    # 4. DELETE — Remove an incident from KB
    # ============================================================
    async def delete(self, incident_id: str) -> bool:
        """Remove an incident from the knowledge base."""
        try:
            self._initialize()
            self._collection.delete(ids=[incident_id])
            logger.info(f"[Retriever] Deleted {incident_id} from KB")
            return True
        except Exception as e:
            logger.error(f"[Retriever] Delete error: {e}")
            return False

    # ============================================================
    # HELPER: Build document text for embedding
    # ============================================================
    def _build_document_text(
        self,
        title: str,
        description: str,
        root_cause: Optional[str],
        resolution: Optional[str],
        affected_systems: List[str]
    ) -> str:
        """
        Build a rich text document to embed.
        More context = better search quality.

        We structure it clearly so the AI can also read it directly.
        """
        parts = [
            f"INCIDENT TITLE: {title}",
            f"DESCRIPTION: {description}",
        ]

        if affected_systems:
            parts.append(f"AFFECTED SYSTEMS: {', '.join(affected_systems)}")

        if root_cause:
            parts.append(f"ROOT CAUSE: {root_cause}")

        if resolution:
            parts.append(f"RESOLUTION: {resolution}")

        return "\n\n".join(parts)

    # ============================================================
    # 5. SEED SAMPLE DATA — for testing and demos
    # ============================================================
    async def seed_sample_incidents(self):
        """
        Add sample historical incidents to the knowledge base.
        Useful for demos and testing — so the RAG has something to search.

        In production, these would come from your real incident history.
        """
        sample_incidents = [
            {
                "incident_id": "SAMPLE-001",
                "title": "Database connection pool exhausted",
                "description": (
                    "Production database connection pool reached maximum capacity. "
                    "All new database queries were timing out. "
                    "Application pods started crashing with connection refused errors."
                ),
                "severity": "P1",
                "affected_systems": ["api-service", "database", "user-service"],
                "root_cause": (
                    "A slow query introduced in deployment v1.2.3 was holding "
                    "connections open for 30+ seconds instead of releasing them. "
                    "Combined with a traffic spike, this exhausted the pool."
                ),
                "resolution": (
                    "Rolled back deployment. Added query timeout of 5 seconds. "
                    "Increased connection pool size from 20 to 50 as temporary fix."
                )
            },
            {
                "incident_id": "SAMPLE-002",
                "title": "Payment service returning 502 errors after deployment",
                "description": (
                    "Payment service began returning 502 Bad Gateway errors "
                    "immediately after deployment of v2.1.0. "
                    "Checkout flow completely broken. Revenue impact estimated at $50k/hour."
                ),
                "severity": "P1",
                "affected_systems": ["payment-service", "checkout-api", "gateway"],
                "root_cause": (
                    "New environment variable PAYMENT_API_KEY was not set in production "
                    "despite being required by the new code. "
                    "The service started but crashed on first payment attempt."
                ),
                "resolution": (
                    "Set missing environment variable in production. "
                    "Added startup health check to verify all required env vars exist. "
                    "Added deployment checklist item for env var verification."
                )
            },
            {
                "incident_id": "SAMPLE-003",
                "title": "Memory leak causing pod OOMKilled every 6 hours",
                "description": (
                    "API pods were being killed by Kubernetes every 6-8 hours "
                    "due to Out of Memory errors. "
                    "Each restart caused 2-3 minutes of degraded performance."
                ),
                "severity": "P2",
                "affected_systems": ["api-service", "kubernetes"],
                "root_cause": (
                    "A background job introduced in the last sprint was caching "
                    "API responses in memory but never evicting old entries. "
                    "Memory grew linearly until the pod hit its 512MB limit."
                ),
                "resolution": (
                    "Added TTL-based cache eviction (30 minute expiry). "
                    "Increased pod memory limit to 1GB as temporary buffer. "
                    "Added memory usage monitoring alert at 80% threshold."
                )
            },
            {
                "incident_id": "SAMPLE-004",
                "title": "Authentication service latency spike — users getting logged out",
                "description": (
                    "Auth service p99 latency spiked from 50ms to 8 seconds. "
                    "Users were experiencing session timeouts and forced logouts. "
                    "Login success rate dropped to 23%."
                ),
                "severity": "P2",
                "affected_systems": ["auth-service", "redis", "user-session"],
                "root_cause": (
                    "Redis cache used for session storage reached memory limit and "
                    "started evicting active sessions. "
                    "Auth service fell back to database for every session lookup, "
                    "overloading the database."
                ),
                "resolution": (
                    "Increased Redis memory from 2GB to 8GB. "
                    "Implemented Redis eviction policy (LRU) to prevent future full-cache crashes. "
                    "Added Redis memory monitoring alert."
                )
            },
            {
                "incident_id": "SAMPLE-005",
                "title": "Data pipeline delayed — reports showing stale data",
                "description": (
                    "Business intelligence reports showing data 18 hours old. "
                    "ETL pipeline had silently failed. "
                    "No alerts fired because the monitoring was checking the wrong metric."
                ),
                "severity": "P3",
                "affected_systems": ["etl-pipeline", "data-warehouse", "reporting"],
                "root_cause": (
                    "Third-party data source API changed its response schema without notice. "
                    "The pipeline was parsing a field that no longer existed, "
                    "silently producing null values and halting processing."
                ),
                "resolution": (
                    "Updated pipeline to handle both old and new API schema. "
                    "Added schema validation step at pipeline ingestion point. "
                    "Fixed monitoring to alert on data freshness (not just job completion)."
                )
            }
        ]

        logger.info(f"[Retriever] Seeding {len(sample_incidents)} sample incidents...")

        seeded = 0
        for incident in sample_incidents:
            success = await self.ingest(**incident)
            if success:
                seeded += 1

        logger.info(f"[Retriever] Seeded {seeded}/{len(sample_incidents)} incidents")
        return seeded


# ============================================================
# Singleton — one instance shared across the app
# ============================================================
# We create one retriever instance and reuse it
# so ChromaDB connection is only made once

from functools import lru_cache
from config import settings

@lru_cache(maxsize=1)
def get_retriever() -> IncidentRetriever:
    """
    Returns the shared IncidentRetriever instance.
    lru_cache ensures only one instance is ever created.

    Usage in other files:
        from rag.retriever import get_retriever
        retriever = get_retriever()
        results = await retriever.search("database timeout")
    """
    return IncidentRetriever(persist_path=settings.CHROMA_PERSIST_PATH)
