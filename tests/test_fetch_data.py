"""Tests for fetch_data.py module."""

from unittest.mock import MagicMock, patch

import pytest


class TestFetchPubmedAbstract:
    """Tests for the fetch_pubmed_abstract function."""

    def test_successful_fetch(self, sample_pubmed_response: dict) -> None:
        """Test successful abstract fetching."""
        with patch("fetch_data.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = sample_pubmed_response
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            # Import here to avoid caching issues
            from fetch_data import fetch_pubmed_abstract

            # Clear cache for testing
            fetch_pubmed_abstract.clear()

            result = fetch_pubmed_abstract("TP53 cancer")

            assert result == "This is a test abstract about TP53 and cancer research."
            mock_get.assert_called_once()

    def test_no_results_found(self) -> None:
        """Test handling of no results."""
        with patch("fetch_data.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"resultList": {"result": []}}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            from fetch_data import fetch_pubmed_abstract

            fetch_pubmed_abstract.clear()

            result = fetch_pubmed_abstract("nonexistent query xyz123")

            assert "No" in result or "found" in result.lower()

    def test_missing_abstract_in_result(self) -> None:
        """Test handling of results without abstractText."""
        with patch("fetch_data.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "resultList": {"result": [{"title": "Test", "id": "123"}]}
            }
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            from fetch_data import fetch_pubmed_abstract

            fetch_pubmed_abstract.clear()

            result = fetch_pubmed_abstract("test query")

            assert "No abstract found" in result

    def test_api_error_handling(self) -> None:
        """Test handling of API errors."""
        with patch("fetch_data.requests.get") as mock_get:
            mock_get.side_effect = Exception("Network error")

            from fetch_data import fetch_pubmed_abstract

            fetch_pubmed_abstract.clear()

            result = fetch_pubmed_abstract("test query")

            assert "Error" in result

    def test_malformed_response_handling(self) -> None:
        """Test handling of malformed API response."""
        with patch("fetch_data.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"unexpected": "format"}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            from fetch_data import fetch_pubmed_abstract

            fetch_pubmed_abstract.clear()

            result = fetch_pubmed_abstract("test query")

            assert "No results found" in result
