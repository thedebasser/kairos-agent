"""Kairos Agent — ChromaDB Knowledge Base for Calibrations.

Wraps ChromaDB HttpClient (connecting to the existing Docker service)
to store and retrieve proven calibration entries.  Uses Ollama's
nomic-embed-text model for embeddings.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from kairos.calibration.models import (
    BASELINE_PHYSICS,
    CalibrationEntry,
    CalibrationMatch,
    CorrectionFactors,
    ScenarioDescriptor,
)
from kairos.calibration.scenario import (
    composite_corrections,
    compute_dimensional_overlap,
)
from kairos.config import get_settings

logger = logging.getLogger(__name__)

# ChromaDB collection name — single collection for all calibrations.
COLLECTION_NAME = "calibrations"


class KnowledgeBase:
    """ChromaDB-backed knowledge base for proven calibrations.

    Uses HttpClient against the existing ChromaDB Docker service
    and Ollama nomic-embed-text for embeddings.
    """

    def __init__(
        self,
        chromadb_host: str | None = None,
        chromadb_port: int | None = None,
        ollama_base_url: str | None = None,
    ) -> None:
        settings = get_settings()
        self._chromadb_host = chromadb_host or settings.chromadb_host
        self._chromadb_port = chromadb_port or settings.chromadb_port
        self._ollama_url = ollama_base_url or settings.ollama_base_url
        self._client: Any | None = None
        self._collection: Any | None = None

    def _get_collection(self) -> Any:
        """Lazy-init ChromaDB client and collection."""
        if self._collection is not None:
            return self._collection

        import chromadb

        self._client = chromadb.HttpClient(
            host=self._chromadb_host,
            port=self._chromadb_port,
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Connected to ChromaDB at %s:%d, collection '%s' (%d entries)",
            self._chromadb_host,
            self._chromadb_port,
            COLLECTION_NAME,
            self._collection.count(),
        )
        return self._collection

    def _embed_text(self, text: str) -> list[float]:
        """Generate embedding via Ollama nomic-embed-text."""
        response = httpx.post(
            f"{self._ollama_url}/api/embed",
            json={"model": "nomic-embed-text", "input": text},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        # Ollama returns {"embeddings": [[...]]} for /api/embed
        embeddings = data.get("embeddings", [])
        if embeddings:
            return embeddings[0]
        msg = f"Ollama embed returned no embeddings: {data}"
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(self, entry: CalibrationEntry) -> None:
        """Store a calibration entry in ChromaDB."""
        collection = self._get_collection()
        doc_id = str(entry.calibration_id)
        document = entry.to_chromadb_document()
        metadata = entry.to_chromadb_metadata()
        embedding = self._embed_text(document)

        collection.upsert(
            ids=[doc_id],
            documents=[document],
            metadatas=[metadata],
            embeddings=[embedding],
        )
        logger.info(
            "Stored calibration %s (confidence=%.2f, type=%s)",
            doc_id, entry.confidence, entry.calibration_type,
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(
        self,
        scenario: ScenarioDescriptor,
        *,
        n_results: int = 5,
        min_confidence: float = 0.5,
        include_unresolved: bool = False,
    ) -> list[CalibrationMatch]:
        """Find similar calibrations for a scenario.

        Returns ranked matches with similarity scores and dimensional overlap.
        """
        collection = self._get_collection()

        if collection.count() == 0:
            return []

        query_text = scenario.to_natural_language()
        embedding = self._embed_text(query_text)

        # Build where filter
        where_conditions: list[dict[str, Any]] = [
            {"confidence": {"$gte": min_confidence}},
        ]
        if not include_unresolved:
            where_conditions.append({"calibration_type": "resolved"})

        where: dict[str, Any]
        if len(where_conditions) == 1:
            where = where_conditions[0]
        else:
            where = {"$and": where_conditions}

        try:
            results = collection.query(
                query_embeddings=[embedding],
                n_results=min(n_results, collection.count()),
                where=where,
            )
        except Exception:
            # If the where filter fails (e.g. no matching entries), try without
            logger.debug("ChromaDB query with filter failed, trying without filter")
            try:
                results = collection.query(
                    query_embeddings=[embedding],
                    n_results=min(n_results, collection.count()),
                )
            except Exception:
                logger.warning("ChromaDB query failed completely", exc_info=True)
                return []

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        matches: list[CalibrationMatch] = []
        for doc_id, meta, distance in zip(
            results["ids"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=True,
        ):
            similarity = 1.0 - distance  # cosine distance → similarity
            overlap, matching_dims = compute_dimensional_overlap(scenario, meta)
            combined_score = (similarity * 0.6) + (overlap * 0.4)

            # Parse stored corrections
            corrections_json = meta.get("corrections_json", "{}")
            try:
                corrections = CorrectionFactors.model_validate_json(corrections_json)
            except Exception:
                corrections = CorrectionFactors()

            matches.append(CalibrationMatch(
                calibration_id=doc_id,
                corrections=corrections,
                similarity=round(similarity, 4),
                dimensional_overlap=overlap,
                combined_score=round(combined_score, 4),
                confidence=meta.get("confidence", 0.0),
                matching_dimensions=matching_dims,
            ))

        return sorted(matches, key=lambda m: m.combined_score, reverse=True)

    def lookup_starting_params(
        self,
        scenario: ScenarioDescriptor,
        *,
        min_confidence: float = 0.5,
        min_combined_score: float = 0.3,
    ) -> CorrectionFactors | None:
        """Look up and composite starting parameters for a scenario.

        Returns composited CorrectionFactors if relevant matches exist,
        or None if the knowledge base has no useful prior information.
        """
        matches = self.lookup(scenario, min_confidence=min_confidence)

        if not matches:
            return None

        # Filter to matches above threshold
        good_matches = [
            m.model_dump() for m in matches if m.combined_score >= min_combined_score
        ]
        if not good_matches:
            return None

        composited = composite_corrections(good_matches, scenario)
        logger.info(
            "Composited corrections from %d matches for %s scenario",
            len(good_matches),
            scenario.path.type.value,
        )
        return composited

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def decay_confidence(self, calibration_id: str, amount: float = 0.05) -> None:
        """Reduce confidence of a calibration that led to failure.

        Called when a stored calibration is retrieved and used but the
        resulting simulation still fails.  Naturally deprioritises
        misleading entries over time.
        """
        collection = self._get_collection()
        try:
            result = collection.get(ids=[calibration_id], include=["metadatas"])
            if result and result["metadatas"]:
                meta = result["metadatas"][0]
                new_confidence = max(0.0, meta.get("confidence", 0.5) - amount)
                meta["confidence"] = new_confidence
                collection.update(ids=[calibration_id], metadatas=[meta])
                logger.info(
                    "Decayed confidence for %s: %.2f → %.2f",
                    calibration_id, meta.get("confidence", 0.5) + amount, new_confidence,
                )
        except Exception:
            logger.warning("Failed to decay confidence for %s", calibration_id, exc_info=True)

    def remove_unresolved(self) -> int:
        """Bulk-remove all unresolved (negative example) entries.

        Returns the count of entries removed.
        """
        collection = self._get_collection()
        try:
            results = collection.get(
                where={"calibration_type": "unresolved"},
                include=[],
            )
            if results and results["ids"]:
                collection.delete(ids=results["ids"])
                count = len(results["ids"])
                logger.info("Removed %d unresolved calibration entries", count)
                return count
        except Exception:
            logger.warning("Failed to remove unresolved entries", exc_info=True)
        return 0

    def count(self) -> int:
        """Return total number of entries in the knowledge base."""
        return self._get_collection().count()
