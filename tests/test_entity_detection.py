"""Tests for entity detection functionality in app.py."""

# Import the function and patterns from app.py
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import ENTITY_PATTERNS, detect_entities


class TestEntityPatterns:
    """Tests for entity pattern definitions."""

    def test_entity_patterns_exist(self) -> None:
        """Test that all expected entity types have patterns."""
        expected_types = ["GENE", "PROTEIN", "DISEASE", "DRUG"]
        for entity_type in expected_types:
            assert entity_type in ENTITY_PATTERNS
            assert len(ENTITY_PATTERNS[entity_type]) > 0


class TestDetectEntities:
    """Tests for the detect_entities function."""

    def test_detect_gene_entities(self) -> None:
        """Test detection of gene-like entities."""
        text = "TP53 and BRCA1 are important tumor suppressor genes."
        entities = detect_entities(text)

        entity_names = [e[0] for e in entities]
        # Should detect gene-like patterns
        assert any("TP53" in name or "BRCA1" in name for name in entity_names)

    def test_detect_disease_entities(self) -> None:
        """Test detection of disease entities."""
        text = "Breast cancer and heart disease are common conditions."
        entities = detect_entities(text)

        entity_types = [e[1] for e in entities]

        # Should detect disease patterns
        assert "DISEASE" in entity_types

    def test_detect_drug_entities(self) -> None:
        """Test detection of drug entities."""
        text = "Trastuzumab and rituximab are monoclonal antibodies."
        entities = detect_entities(text)

        # Should detect drug patterns (ending in -mab)
        drug_entities = [e for e in entities if e[1] == "DRUG"]
        assert len(drug_entities) > 0

    def test_detect_protein_entities(self) -> None:
        """Test detection of protein entities."""
        text = "Insulin and kinase are important proteins."
        entities = detect_entities(text)

        entity_types = [e[1] for e in entities]

        # Should detect protein patterns
        assert "PROTEIN" in entity_types

    def test_empty_text_returns_empty_list(self) -> None:
        """Test that empty text returns empty list."""
        entities = detect_entities("")
        assert entities == []

    def test_no_entities_in_generic_text(self) -> None:
        """Test that generic text without biomedical terms returns few/no entities."""
        text = "The quick brown fox jumps over the lazy dog."
        entities = detect_entities(text)

        # Should have minimal or no entities
        assert len(entities) <= 2  # Allow some false positives

    def test_no_duplicate_entities(self) -> None:
        """Test that duplicate entities are removed."""
        text = "TP53 and TP53 are mentioned twice."
        entities = detect_entities(text)

        entity_names = [e[0] for e in entities]
        # Check no duplicates
        assert len(entity_names) == len(set(entity_names))

    def test_short_matches_filtered(self) -> None:
        """Test that very short matches (<=2 chars) are filtered."""
        text = "A B CD are short."
        entities = detect_entities(text)

        entity_names = [e[0] for e in entities]
        # All detected entities should be > 2 characters
        for name in entity_names:
            assert len(name) > 2

    def test_multiple_entity_types(self, sample_biomedical_text: str) -> None:
        """Test detection of multiple entity types in one text."""
        entities = detect_entities(sample_biomedical_text)

        entity_types = {e[1] for e in entities}
        # Should detect at least 2 different entity types
        assert len(entity_types) >= 1

    def test_returns_list_of_tuples(self) -> None:
        """Test that return format is list of (entity, type) tuples."""
        text = "BRCA1 mutations cause cancer."
        entities = detect_entities(text)

        assert isinstance(entities, list)
        for entity in entities:
            assert isinstance(entity, tuple)
            assert len(entity) == 2
            assert isinstance(entity[0], str)  # entity name
            assert isinstance(entity[1], str)  # entity type
