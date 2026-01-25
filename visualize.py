"""Visualization utilities for attention heatmaps."""

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go


def plot_attention_heatmap(
    tokens: list[str],
    attention_weights: npt.NDArray[np.float32],
    title: str = "Attention Heatmap",
) -> go.Figure:
    """Create interactive attention heatmap using Plotly"""
    # Filter out padding tokens
    filtered_tokens = []
    filtered_attention = []

    # Check if tokens and attention_weights are valid
    if not tokens or attention_weights.size == 0:
        return go.Figure().update_layout(
            title="No attention data available",
            annotations=[
                {
                    "text": "Unable to visualize: No valid attention data",
                    "showarrow": False,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                }
            ],
        )

    # Ensure we only process as many tokens as we have attention weights for
    max_tokens = min(len(tokens), attention_weights.shape[0])

    for i in range(max_tokens):
        token = tokens[i]
        if token not in ["[PAD]", "<pad>"]:
            filtered_tokens.append(token)
            # Get attention for this token, but only up to the number of filtered tokens
            # to ensure a square matrix
            row_data = attention_weights[i, :max_tokens]
            # Only keep attention to non-padding tokens
            row = []
            for j in range(len(row_data)):
                if j < len(filtered_tokens):
                    row.append(row_data[j])
            filtered_attention.append(row)

    # If we have no tokens left after filtering, return empty figure
    if not filtered_tokens:
        return go.Figure().update_layout(
            title="No non-padding tokens found",
            annotations=[
                {
                    "text": "Unable to visualize: All tokens were padding tokens",
                    "showarrow": False,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                }
            ],
        )

    # Ensure filtered_attention is a rectangular array
    max_len = max(len(row) for row in filtered_attention)
    for i in range(len(filtered_attention)):
        while len(filtered_attention[i]) < max_len:
            filtered_attention[i].append(0)  # Pad with zeros

    # Convert to numpy array
    filtered_attention = np.array(filtered_attention)

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=filtered_attention,
            x=filtered_tokens,
            y=filtered_tokens,
            colorscale="Viridis",
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=title,
        height=600,
        width=700,
        xaxis_title="Tokens",
        yaxis_title="Tokens",
        xaxis={"side": "top"},
    )

    return fig
