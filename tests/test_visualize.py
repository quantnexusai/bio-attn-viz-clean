"""Tests for visualize.py module."""

import numpy as np
import plotly.graph_objects as go
import pytest

from visualize import plot_attention_heatmap


class TestPlotAttentionHeatmap:
    """Tests for the plot_attention_heatmap function."""

    def test_basic_heatmap_creation(
        self, sample_tokens: list[str], sample_attention_weights: np.ndarray
    ) -> None:
        """Test that a basic heatmap is created successfully."""
        fig = plot_attention_heatmap(sample_tokens, sample_attention_weights)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.data[0].type == "heatmap"

    def test_heatmap_with_custom_title(
        self, sample_tokens: list[str], sample_attention_weights: np.ndarray
    ) -> None:
        """Test heatmap with custom title."""
        custom_title = "Custom Attention Title"
        fig = plot_attention_heatmap(
            sample_tokens, sample_attention_weights, title=custom_title
        )

        assert fig.layout.title.text == custom_title

    def test_empty_tokens_returns_empty_figure(
        self, sample_attention_weights: np.ndarray
    ) -> None:
        """Test that empty tokens list returns an empty figure with message."""
        fig = plot_attention_heatmap([], sample_attention_weights)

        assert isinstance(fig, go.Figure)
        # Should have an annotation explaining the issue
        assert len(fig.layout.annotations) > 0

    def test_empty_attention_returns_empty_figure(
        self, sample_tokens: list[str], empty_attention_weights: np.ndarray
    ) -> None:
        """Test that empty attention weights returns an empty figure."""
        fig = plot_attention_heatmap(sample_tokens, empty_attention_weights)

        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) > 0

    def test_padding_tokens_filtered(self) -> None:
        """Test that padding tokens are filtered out."""
        tokens = ["[CLS]", "hello", "[PAD]", "[PAD]", "[SEP]"]
        weights = np.random.rand(5, 5).astype(np.float32)
        weights = weights / weights.sum(axis=1, keepdims=True)

        fig = plot_attention_heatmap(tokens, weights)

        # The heatmap should not include [PAD] tokens
        if fig.data and hasattr(fig.data[0], "x") and fig.data[0].x is not None:
            x_labels = list(fig.data[0].x)
            assert "[PAD]" not in x_labels

    def test_heatmap_dimensions(
        self, sample_tokens: list[str], sample_attention_weights: np.ndarray
    ) -> None:
        """Test that heatmap has correct dimensions."""
        fig = plot_attention_heatmap(sample_tokens, sample_attention_weights)

        assert fig.layout.height == 600
        assert fig.layout.width == 700

    def test_colorscale_is_viridis(
        self, sample_tokens: list[str], sample_attention_weights: np.ndarray
    ) -> None:
        """Test that the colorscale is Viridis."""
        fig = plot_attention_heatmap(sample_tokens, sample_attention_weights)

        assert fig.data[0].colorscale == "Viridis"
