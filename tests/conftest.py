"""Pytest fixtures for testing."""

import numpy as np
import pytest


@pytest.fixture
def sample_tokens() -> list[str]:
    """Sample tokens for testing."""
    return ["[CLS]", "TP", "##53", "mutations", "are", "common", "[SEP]"]


@pytest.fixture
def sample_attention_weights() -> np.ndarray:
    """Sample attention weight matrix for testing."""
    size = 7
    # Create a random attention matrix (normalized rows)
    weights = np.random.rand(size, size)
    # Normalize each row to sum to 1 (like real attention weights)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return weights.astype(np.float32)


@pytest.fixture
def empty_attention_weights() -> np.ndarray:
    """Empty attention weights for edge case testing."""
    return np.array([])


@pytest.fixture
def sample_biomedical_text() -> str:
    """Sample biomedical text for testing."""
    return "TP53 mutations are associated with breast cancer and tumor suppression."


@pytest.fixture
def sample_pubmed_response() -> dict:
    """Sample PubMed API response for testing."""
    return {
        "resultList": {
            "result": [
                {
                    "abstractText": "This is a test abstract about TP53 and cancer research.",
                    "title": "Test Article",
                    "id": "12345",
                }
            ]
        }
    }
