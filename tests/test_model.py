"""Tests for model.py module.

Note: These tests use mocking to avoid loading the actual BioBERT model,
which would be slow and require significant memory/disk space.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import torch


class TestBioBERTAttention:
    """Tests for the BioBERTAttention class."""

    @patch("model.AutoModel")
    @patch("model.AutoTokenizer")
    def test_model_initialization(
        self, mock_tokenizer_class: MagicMock, mock_model_class: MagicMock
    ) -> None:
        """Test that model initializes correctly."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        from model import BioBERTAttention

        BioBERTAttention()  # noqa: F841

        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()
        mock_model.eval.assert_called_once()

    @patch("model.AutoModel")
    @patch("model.AutoTokenizer")
    def test_custom_model_name(
        self, mock_tokenizer_class: MagicMock, mock_model_class: MagicMock
    ) -> None:
        """Test initialization with custom model name."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        from model import BioBERTAttention

        custom_model = "custom/model-name"
        BioBERTAttention(model_name=custom_model)  # noqa: F841

        mock_tokenizer_class.from_pretrained.assert_called_with(custom_model)

    @patch("model.AutoModel")
    @patch("model.AutoTokenizer")
    def test_get_attention_weights(
        self, mock_tokenizer_class: MagicMock, mock_model_class: MagicMock
    ) -> None:
        """Test get_attention_weights method."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        # Mock tokenizer behavior
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 1234, 5678, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }
        mock_tokenizer.convert_ids_to_tokens.return_value = [
            "[CLS]",
            "test",
            "text",
            "[SEP]",
        ]

        # Mock model output with attention
        mock_attention = torch.rand(1, 12, 4, 4)  # batch, heads, seq, seq
        mock_outputs = MagicMock()
        mock_outputs.attentions = [mock_attention] * 12  # 12 layers
        mock_model.return_value = mock_outputs

        from model import BioBERTAttention

        bio_bert = BioBERTAttention()
        tokens, attentions = bio_bert.get_attention_weights("test text", layer=0, head=0)

        assert isinstance(tokens, list)
        assert len(tokens) == 4
        assert isinstance(attentions, np.ndarray)
        assert attentions.shape == (4, 4)

    @patch("model.AutoModel")
    @patch("model.AutoTokenizer")
    def test_get_average_attention(
        self, mock_tokenizer_class: MagicMock, mock_model_class: MagicMock
    ) -> None:
        """Test get_average_attention method."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        # Mock tokenizer behavior
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 1234, 5678, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }
        mock_tokenizer.convert_ids_to_tokens.return_value = [
            "[CLS]",
            "test",
            "text",
            "[SEP]",
        ]

        # Mock model output with attention
        mock_attention = torch.rand(1, 12, 4, 4)  # batch, heads, seq, seq
        mock_outputs = MagicMock()
        mock_outputs.attentions = [mock_attention] * 12  # 12 layers
        mock_model.return_value = mock_outputs

        from model import BioBERTAttention

        bio_bert = BioBERTAttention()
        tokens, attentions = bio_bert.get_average_attention("test text")

        assert isinstance(tokens, list)
        assert len(tokens) == 4
        assert isinstance(attentions, np.ndarray)
        assert attentions.shape == (4, 4)

    @patch("model.AutoModel")
    @patch("model.AutoTokenizer")
    def test_layer_and_head_selection(
        self, mock_tokenizer_class: MagicMock, mock_model_class: MagicMock
    ) -> None:
        """Test that different layers and heads can be selected."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_tokenizer.convert_ids_to_tokens.return_value = ["[CLS]", "[SEP]"]

        # Create distinct attention patterns for each layer
        mock_attentions = []
        for layer in range(12):
            layer_attention = torch.zeros(1, 12, 2, 2)
            layer_attention[0, :, :, :] = layer * 0.01  # Different value per layer
            mock_attentions.append(layer_attention)

        mock_outputs = MagicMock()
        mock_outputs.attentions = mock_attentions
        mock_model.return_value = mock_outputs

        from model import BioBERTAttention

        bio_bert = BioBERTAttention()

        # Test different layers
        _, attn_layer_0 = bio_bert.get_attention_weights("test", layer=0, head=0)
        _, attn_layer_5 = bio_bert.get_attention_weights("test", layer=5, head=0)

        # Attention values should differ between layers
        assert not np.allclose(attn_layer_0, attn_layer_5)
