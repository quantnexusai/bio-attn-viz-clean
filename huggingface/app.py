"""
BioBERT Attention Visualization - Gradio Interface with MCP Support
Analyze how BioBERT attends to biomedical text via AI assistants.
"""

import json
import re

import gradio as gr
import numpy as np
import requests

# Entity detection patterns
ENTITY_PATTERNS: dict[str, list[str]] = {
    "GENE": [r"\b[A-Z][A-Z0-9]{1,5}\b", r"\b[A-Z][a-z]{1,}[0-9]\b"],
    "PROTEIN": [r"\b[A-Z][a-z]{2,}(in|ase)\b", r"\b[A-Z][a-z]+[A-Z][a-z]+\b"],
    "DISEASE": [r"\b[Cc]ancer\b", r"\b[Dd]isease\b", r"\b[Tt]umou?r\b", r"\b[Ss]yndrome\b"],
    "DRUG": [r"\b[A-Za-z]+(mab|zumab|ximab)\b", r"\b[A-Za-z]+(olol|asone|idine)\b"],
}

# Lazy load model to avoid startup delays
_model = None


def get_model():
    """Lazy load BioBERT model."""
    global _model
    if _model is None:
        from transformers import AutoModel, AutoTokenizer

        model_name = "dmis-lab/biobert-base-cased-v1.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, output_attentions=True)
        model.eval()
        _model = (tokenizer, model)
    return _model


def detect_entities(text: str) -> list[dict]:
    """Detect biomedical entities in text using pattern matching."""
    entities = []
    seen = set()

    for entity_type, patterns in ENTITY_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                entity = match.group()
                if len(entity) > 2 and entity not in seen:
                    seen.add(entity)
                    entities.append({"name": entity, "type": entity_type})

    return entities


def fetch_pubmed_abstract(query: str) -> str:
    """Fetch abstract from PubMed via EuropePMC API."""
    api_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {"query": query, "format": "json", "resultType": "core", "pageSize": 1}

    try:
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "resultList" in data and "result" in data["resultList"]:
            results = data["resultList"]["result"]
            for result in results:
                if "abstractText" in result:
                    return result["abstractText"]
        return "No abstract found for this query."
    except Exception as e:
        return f"Error fetching abstract: {e}"


def analyze_attention(text: str, layer: int = 0, head: int = 0, use_average: bool = False) -> str:
    """
    Analyze BioBERT attention patterns for biomedical text.

    Args:
        text: Biomedical text to analyze
        layer: Transformer layer (0-11)
        head: Attention head (0-11)
        use_average: If True, average across all layers and heads

    Returns:
        JSON with tokens, attention matrix, and detected entities
    """
    if not text or not text.strip():
        return json.dumps({"error": "Please provide text to analyze"}, indent=2)

    text = text.strip()

    try:
        import torch

        tokenizer, model = get_model()

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Get attention weights
        with torch.no_grad():
            outputs = model(**inputs)

            if use_average:
                attention = torch.stack(outputs.attentions).mean(dim=0).mean(dim=1)[0].numpy()
                title = "Average Attention (all layers & heads)"
            else:
                layer = max(0, min(11, layer))
                head = max(0, min(11, head))
                attention = outputs.attentions[layer][0][head].numpy()
                title = f"Attention (Layer {layer}, Head {head})"

        # Filter padding tokens
        filtered_tokens = [t for t in tokens if t not in ["[PAD]", "<pad>", "[CLS]", "[SEP]"]]
        n_tokens = len(filtered_tokens)

        # Get attention for non-special tokens (skip [CLS] at start, [SEP] at end)
        if len(tokens) > 2:
            attention_subset = attention[1:-1, 1:-1]  # Remove CLS/SEP
            attention_subset = attention_subset[:n_tokens, :n_tokens]
        else:
            attention_subset = attention

        # Detect entities
        entities = detect_entities(text)

        # Find high-attention pairs (top 10 for better insights)
        high_attention_pairs = []
        if attention_subset.size > 0:
            flat_indices = np.argsort(attention_subset.flatten())[-10:][::-1]
            for idx in flat_indices:
                i, j = divmod(idx, attention_subset.shape[1])
                if i < len(filtered_tokens) and j < len(filtered_tokens):
                    high_attention_pairs.append(
                        {
                            "from": filtered_tokens[i],
                            "to": filtered_tokens[j],
                            "score": round(float(attention_subset[i, j]), 4),
                        }
                    )

        # Return compact summary (no full matrix - too large for AI assistants)
        result = {
            "success": True,
            "analysis": title,
            "num_tokens": len(filtered_tokens),
            "entities": entities,
            "top_attention_pairs": high_attention_pairs,
            "stats": {
                "mean": round(float(attention_subset.mean()), 4),
                "max": round(float(attention_subset.max()), 4),
            },
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


def process_pubmed_query(query: str, layer: int, head: int, use_average: bool) -> tuple[str, str]:
    """Fetch PubMed abstract and analyze attention."""
    abstract = fetch_pubmed_abstract(query)
    if abstract.startswith("Error") or abstract.startswith("No abstract"):
        return abstract, json.dumps({"error": abstract}, indent=2)

    analysis = analyze_attention(abstract, layer, head, use_average)
    return abstract, analysis


def analyze_text_direct(text: str, layer: int, head: int, use_average: bool) -> str:
    """Direct text analysis wrapper for Gradio."""
    return analyze_attention(text, layer, head, use_average)


# Create Gradio interface
with gr.Blocks(title="BioBERT Attention Analyzer") as demo:
    gr.Markdown(
        """
# BioBERT Attention Analyzer

Visualize and analyze how BioBERT attends to biomedical text.
Explore attention patterns across 12 layers and 12 attention heads.

[GitHub](https://github.com/quantnexusai/bio-attn-viz-clean) |
[Streamlit Demo](https://bio-attn-viz.streamlit.app/)
    """
    )

    with gr.Tab("Analyze Text"):
        text_input = gr.Textbox(
            label="Biomedical Text",
            placeholder="Enter biomedical text (e.g., 'TP53 mutations are associated with breast cancer.')",
            lines=4,
            value="TP53 mutations are associated with breast cancer progression and poor prognosis.",
        )

        with gr.Row():
            layer_slider = gr.Slider(0, 11, value=0, step=1, label="Layer")
            head_slider = gr.Slider(0, 11, value=0, step=1, label="Attention Head")
            avg_checkbox = gr.Checkbox(label="Average Attention", value=False)

        analyze_btn = gr.Button("Analyze Attention", variant="primary")

        output_json = gr.Code(label="Analysis Results", language="json")

        analyze_btn.click(
            fn=analyze_text_direct,
            inputs=[text_input, layer_slider, head_slider, avg_checkbox],
            outputs=output_json,
        )

    with gr.Tab("PubMed Search"):
        query_input = gr.Textbox(
            label="PubMed Query",
            placeholder="Search PubMed (e.g., 'BRCA1 breast cancer')",
            value="TP53 breast cancer",
        )

        with gr.Row():
            pm_layer = gr.Slider(0, 11, value=0, step=1, label="Layer")
            pm_head = gr.Slider(0, 11, value=0, step=1, label="Attention Head")
            pm_avg = gr.Checkbox(label="Average Attention", value=False)

        fetch_btn = gr.Button("Fetch & Analyze", variant="primary")

        abstract_output = gr.Textbox(label="Fetched Abstract", lines=6)
        pm_json_output = gr.Code(label="Analysis Results", language="json")

        fetch_btn.click(
            fn=process_pubmed_query,
            inputs=[query_input, pm_layer, pm_head, pm_avg],
            outputs=[abstract_output, pm_json_output],
        )

    with gr.Tab("Entity Detection"):
        entity_input = gr.Textbox(
            label="Text for Entity Detection",
            placeholder="Enter biomedical text...",
            lines=4,
            value="The BRCA1 gene mutations increase breast cancer risk. Trastuzumab is used for HER2-positive tumors.",
        )

        detect_btn = gr.Button("Detect Entities", variant="primary")

        entity_output = gr.Code(label="Detected Entities", language="json")

        def detect_entities_json(text):
            entities = detect_entities(text)
            return json.dumps({"entities": entities, "count": len(entities)}, indent=2)

        detect_btn.click(fn=detect_entities_json, inputs=entity_input, outputs=entity_output)

    gr.Markdown(
        """
---
**About BioBERT Attention Analyzer**

This tool helps researchers understand how BioBERT (a domain-specific BERT model)
processes biomedical text by visualizing attention patterns.

**Features:**
- Analyze attention from any of 12 layers × 12 heads
- Fetch and analyze PubMed abstracts directly
- Detect biomedical entities (genes, proteins, diseases, drugs)

MIT License | Built with PyTorch, Transformers, and Gradio
    """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # nosec B104
        server_port=7860,
        mcp_server=True,
    )
