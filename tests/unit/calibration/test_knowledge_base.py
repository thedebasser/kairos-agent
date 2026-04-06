"""Unit tests for calibration knowledge base.

Mocks ChromaDB client and Ollama embedding endpoint to test
store, lookup, decay_confidence, and remove_unresolved logic
without any external services.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from kairos.calibration.knowledge_base import COLLECTION_NAME, KnowledgeBase
from kairos.calibration.models import (
    CalibrationEntry,
    CorrectionFactors,
    PathDescriptor,
    PathType,
    ScenarioDescriptor,
)


pytestmark = pytest.mark.unit


# =============================================================================
# Helpers
# =============================================================================

FAKE_EMBEDDING = [0.1] * 768


def _make_entry(
    *,
    path_type: PathType = PathType.STRAIGHT,
    domino_count: int = 30,
    confidence: float = 0.85,
    spacing_ratio: float = 0.92,
) -> CalibrationEntry:
    scenario = ScenarioDescriptor(
        path=PathDescriptor(type=path_type, amplitude=0.0, cycles=1.0),
        domino_count=domino_count,
    )
    return CalibrationEntry(
        scenario=scenario,
        corrections=CorrectionFactors(spacing_ratio=spacing_ratio),
        confidence=confidence,
        iteration_count=3,
    )


def _make_scenario(
    path_type: PathType = PathType.STRAIGHT,
    domino_count: int = 30,
) -> ScenarioDescriptor:
    return ScenarioDescriptor(
        path=PathDescriptor(type=path_type, amplitude=0.0, cycles=1.0),
        domino_count=domino_count,
    )


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_collection() -> MagicMock:
    """A mock ChromaDB collection."""
    coll = MagicMock()
    coll.count.return_value = 5
    return coll


@pytest.fixture
def kb(mock_collection: MagicMock) -> KnowledgeBase:
    """KnowledgeBase with mocked ChromaDB and Ollama."""
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._chromadb_host = "localhost"
    kb._chromadb_port = 8000
    kb._ollama_url = "http://localhost:11434"
    kb._client = MagicMock()
    kb._collection = mock_collection
    return kb


@pytest.fixture
def _mock_embed(kb: KnowledgeBase) -> None:
    """Patch _embed_text on the kb instance."""
    kb._embed_text = MagicMock(return_value=FAKE_EMBEDDING)  # type: ignore[method-assign]


# =============================================================================
# store()
# =============================================================================

class TestStore:
    @pytest.mark.usefixtures("_mock_embed")
    def test_store_calls_upsert(
        self, kb: KnowledgeBase, mock_collection: MagicMock
    ) -> None:
        entry = _make_entry()
        kb.store(entry)

        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args
        ids = call_kwargs[1]["ids"] if call_kwargs[1] else call_kwargs[0][0]
        assert str(entry.calibration_id) in ids[0] if isinstance(ids, list) else True

    @pytest.mark.usefixtures("_mock_embed")
    def test_store_generates_embedding(self, kb: KnowledgeBase) -> None:
        entry = _make_entry()
        kb.store(entry)
        kb._embed_text.assert_called_once()  # type: ignore[attr-defined]

    @pytest.mark.usefixtures("_mock_embed")
    def test_store_metadata_contains_corrections_json(
        self, kb: KnowledgeBase, mock_collection: MagicMock
    ) -> None:
        entry = _make_entry(spacing_ratio=0.92)
        kb.store(entry)

        call_kwargs = mock_collection.upsert.call_args[1]
        meta = call_kwargs["metadatas"][0]
        assert "corrections_json" in meta
        assert "0.92" in meta["corrections_json"]


# =============================================================================
# lookup()
# =============================================================================

class TestLookup:
    @pytest.mark.usefixtures("_mock_embed")
    def test_lookup_empty_collection_returns_empty(
        self, kb: KnowledgeBase, mock_collection: MagicMock
    ) -> None:
        mock_collection.count.return_value = 0
        result = kb.lookup(_make_scenario())
        assert result == []

    @pytest.mark.usefixtures("_mock_embed")
    def test_lookup_returns_sorted_matches(
        self, kb: KnowledgeBase, mock_collection: MagicMock
    ) -> None:
        entry = _make_entry()
        corrections_json = CorrectionFactors(spacing_ratio=0.92).model_dump_json()
        mock_collection.query.return_value = {
            "ids": [["id-1", "id-2"]],
            "metadatas": [[
                {
                    "confidence": 0.9,
                    "calibration_type": "resolved",
                    "corrections_json": corrections_json,
                    "path_type": "straight",
                    "surface_type": "flat",
                    "size_profile": "uniform",
                    "domino_count": 30,
                    "domino_height": 0.4,
                },
                {
                    "confidence": 0.7,
                    "calibration_type": "resolved",
                    "corrections_json": CorrectionFactors().model_dump_json(),
                    "path_type": "spiral",
                    "surface_type": "flat",
                    "size_profile": "uniform",
                    "domino_count": 200,
                    "domino_height": 0.4,
                },
            ]],
            "distances": [[0.1, 0.5]],
        }

        matches = kb.lookup(_make_scenario())
        assert len(matches) == 2
        # First match should have higher combined_score
        assert matches[0].combined_score >= matches[1].combined_score

    @pytest.mark.usefixtures("_mock_embed")
    def test_lookup_parses_correction_factors(
        self, kb: KnowledgeBase, mock_collection: MagicMock
    ) -> None:
        corrections_json = CorrectionFactors(spacing_ratio=0.88).model_dump_json()
        mock_collection.query.return_value = {
            "ids": [["id-1"]],
            "metadatas": [[{
                "confidence": 0.9,
                "calibration_type": "resolved",
                "corrections_json": corrections_json,
                "path_type": "straight",
                "surface_type": "flat",
                "size_profile": "uniform",
                "domino_count": 30,
                "domino_height": 0.4,
            }]],
            "distances": [[0.15]],
        }

        matches = kb.lookup(_make_scenario())
        assert len(matches) == 1
        assert matches[0].corrections.spacing_ratio == pytest.approx(0.88)

    @pytest.mark.usefixtures("_mock_embed")
    def test_lookup_fallback_on_filter_failure(
        self, kb: KnowledgeBase, mock_collection: MagicMock
    ) -> None:
        """When filtered query raises, fallback to unfiltered query."""
        corrections_json = CorrectionFactors().model_dump_json()
        mock_collection.query.side_effect = [
            RuntimeError("filter failed"),
            {
                "ids": [["id-1"]],
                "metadatas": [[{
                    "confidence": 0.6,
                    "corrections_json": corrections_json,
                    "path_type": "straight",
                    "surface_type": "flat",
                    "size_profile": "uniform",
                    "domino_count": 30,
                    "domino_height": 0.4,
                }]],
                "distances": [[0.2]],
            },
        ]

        matches = kb.lookup(_make_scenario())
        assert len(matches) == 1
        assert mock_collection.query.call_count == 2

    @pytest.mark.usefixtures("_mock_embed")
    def test_lookup_returns_empty_on_total_failure(
        self, kb: KnowledgeBase, mock_collection: MagicMock
    ) -> None:
        mock_collection.query.side_effect = RuntimeError("total failure")
        matches = kb.lookup(_make_scenario())
        assert matches == []


# =============================================================================
# lookup_starting_params()
# =============================================================================

class TestLookupStartingParams:
    @pytest.mark.usefixtures("_mock_embed")
    def test_returns_none_when_no_matches(
        self, kb: KnowledgeBase, mock_collection: MagicMock
    ) -> None:
        mock_collection.count.return_value = 0
        result = kb.lookup_starting_params(_make_scenario())
        assert result is None

    @pytest.mark.usefixtures("_mock_embed")
    def test_returns_composited_corrections(
        self, kb: KnowledgeBase, mock_collection: MagicMock
    ) -> None:
        corrections_json = CorrectionFactors(spacing_ratio=0.9).model_dump_json()
        mock_collection.query.return_value = {
            "ids": [["id-1"]],
            "metadatas": [[{
                "confidence": 0.9,
                "calibration_type": "resolved",
                "corrections_json": corrections_json,
                "path_type": "straight",
                "surface_type": "flat",
                "size_profile": "uniform",
                "domino_count": 30,
                "domino_height": 0.4,
            }]],
            "distances": [[0.05]],
        }

        result = kb.lookup_starting_params(_make_scenario())
        assert result is not None
        assert isinstance(result, CorrectionFactors)


# =============================================================================
# decay_confidence()
# =============================================================================

class TestDecayConfidence:
    def test_decay_reduces_confidence(
        self, kb: KnowledgeBase, mock_collection: MagicMock
    ) -> None:
        mock_collection.get.return_value = {
            "metadatas": [{"confidence": 0.8}],
        }
        kb.decay_confidence("cal-123", amount=0.1)

        mock_collection.update.assert_called_once()
        call_kwargs = mock_collection.update.call_args[1]
        new_meta = call_kwargs["metadatas"][0]
        assert new_meta["confidence"] == pytest.approx(0.7)

    def test_decay_floors_at_zero(
        self, kb: KnowledgeBase, mock_collection: MagicMock
    ) -> None:
        mock_collection.get.return_value = {
            "metadatas": [{"confidence": 0.02}],
        }
        kb.decay_confidence("cal-123", amount=0.05)

        call_kwargs = mock_collection.update.call_args[1]
        new_meta = call_kwargs["metadatas"][0]
        assert new_meta["confidence"] == pytest.approx(0.0)

    def test_decay_handles_missing_entry(
        self, kb: KnowledgeBase, mock_collection: MagicMock
    ) -> None:
        mock_collection.get.return_value = {"metadatas": []}
        # Should not raise
        kb.decay_confidence("nonexistent")
        mock_collection.update.assert_not_called()


# =============================================================================
# remove_unresolved()
# =============================================================================

class TestRemoveUnresolved:
    def test_removes_unresolved_entries(
        self, kb: KnowledgeBase, mock_collection: MagicMock
    ) -> None:
        mock_collection.get.return_value = {"ids": ["id-1", "id-2"]}
        removed = kb.remove_unresolved()

        mock_collection.delete.assert_called_once_with(ids=["id-1", "id-2"])
        assert removed == 2

    def test_returns_zero_when_none_found(
        self, kb: KnowledgeBase, mock_collection: MagicMock
    ) -> None:
        mock_collection.get.return_value = {"ids": []}
        removed = kb.remove_unresolved()
        assert removed == 0
        mock_collection.delete.assert_not_called()


# =============================================================================
# count()
# =============================================================================

class TestCount:
    def test_count_delegates_to_collection(
        self, kb: KnowledgeBase, mock_collection: MagicMock
    ) -> None:
        mock_collection.count.return_value = 42
        assert kb.count() == 42


# =============================================================================
# _embed_text() (with mocked httpx)
# =============================================================================

class TestEmbedText:
    def test_embed_calls_ollama_endpoint(self, kb: KnowledgeBase) -> None:
        fake_response = MagicMock()
        fake_response.json.return_value = {"embeddings": [FAKE_EMBEDDING]}
        fake_response.raise_for_status = MagicMock()

        with patch("kairos.calibration.knowledge_base.httpx.post", return_value=fake_response) as mock_post:
            result = kb._embed_text("test text")

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "/api/embed" in call_args[0][0]
        assert call_args[1]["json"]["model"] == "nomic-embed-text"
        assert result == FAKE_EMBEDDING

    def test_embed_raises_on_empty_embeddings(self, kb: KnowledgeBase) -> None:
        fake_response = MagicMock()
        fake_response.json.return_value = {"embeddings": []}
        fake_response.raise_for_status = MagicMock()

        with patch("kairos.calibration.knowledge_base.httpx.post", return_value=fake_response):
            with pytest.raises(ValueError, match="no embeddings"):
                kb._embed_text("test text")
