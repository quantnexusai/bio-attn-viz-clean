---
title: BioBERT Attention Analyzer
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: Analyze BioBERT attention patterns on biomedical text
---

# BioBERT Attention Analyzer

Analyze how BioBERT attends to biomedical text. Explore attention patterns across 12 layers and 12 attention heads.

## Features

- **Attention Analysis**: Extract attention weights from any layer/head combination
- **PubMed Integration**: Fetch abstracts directly from PubMed for analysis
- **Entity Detection**: Identify genes, proteins, diseases, and drugs
- **MCP Server**: Use with Claude and other AI assistants

## MCP Integration

This Space runs as an MCP server. Add it to Claude:

```
https://quantnexusai-bio-attn-viz.hf.space/gradio_api/mcp/sse
```

## Example Queries

- "How does BioBERT attend to 'TP53' in this text about cancer?"
- "Analyze the attention pattern for BRCA1 gene mutations"
- "What biomedical entities are in this abstract?"

## Links

- [GitHub Repository](https://github.com/quantnexusai/bio-attn-viz-clean)
- [Streamlit Demo](https://bio-attn-viz.streamlit.app/)
