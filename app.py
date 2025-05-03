import streamlit as st
import re
import numpy as np
from model import BioBERTAttention
from visualize import plot_attention_heatmap
from fetch_data import fetch_pubmed_abstract

# Simple entity detection patterns
ENTITY_PATTERNS = {
    "GENE": [r'\b[A-Z][A-Z0-9]{1,5}\b', r'\b[A-Z][a-z]{1,}[0-9]\b'],
    "PROTEIN": [r'\b[A-Z][a-z]{2,}(in|ase)\b', r'\b[A-Z][a-z]+[A-Z][a-z]+\b'],
    "DISEASE": [r'\b[Cc]ancer\b', r'\b[Dd]isease\b', r'\b[Tt]umou?r\b', r'\b[Ss]yndrome\b'],
    "DRUG": [r'\b[A-Za-z]+(mab|zumab|ximab)\b', r'\b[A-Za-z]+(olol|asone|idine)\b']
}

def detect_entities(text):
    """Simple rule-based entity detection"""
    entities = []
    
    for entity_type, patterns in ENTITY_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                if len(match.group()) > 2:  # Filter out short matches
                    entities.append((match.group(), entity_type))
    
    # Remove duplicates
    unique_entities = []
    seen = set()
    for entity, entity_type in entities:
        if entity not in seen:
            seen.add(entity)
            unique_entities.append((entity, entity_type))
    
    return unique_entities

# Cache the model loading to improve performance
@st.cache_resource
def load_model():
    return BioBERTAttention()

def main():
    st.title("BioBERT Attention Visualization Tool")
    st.write("Explore attention patterns in BioBERT for biomedical text.")
    
    # Initialize model
    with st.spinner("Loading BioBERT model..."):
        try:
            model = load_model()
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()
    
    # Input options
    st.subheader("Input Text")
    input_option = st.radio("Choose Input Method:", ["Enter Text", "Fetch from PubMed"])
    
    if input_option == "Enter Text":
        text = st.text_area("Enter biomedical text:", "TP53 mutations are associated with breast cancer.")
    else:
        query = st.text_input("PubMed Search Query:", "TP53 breast cancer")
        if st.button("Fetch Abstract"):
            text = fetch_pubmed_abstract(query)
            st.text_area("Fetched Abstract:", text, height=200)
        else:
            text = "TP53 mutations are associated with breast cancer."
    
    # Visualization settings
    st.subheader("Visualization Settings")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        layer = st.slider("Layer:", 0, 11, 0)
    with col2:
        head = st.slider("Attention Head:", 0, 11, 0)
    with col3:
        avg_attention = st.checkbox("Show Average Attention")
    
    # Visualization
    if st.button("Visualize Attention"):
        with st.spinner("Generating visualization..."):
            try:
                if avg_attention:
                    tokens, attentions = model.get_average_attention(text)
                    title = "Average Attention Across All Layers and Heads"
                else:
                    tokens, attentions = model.get_attention_weights(text, layer=layer, head=head)
                    title = f"Attention (Layer {layer}, Head {head})"
                
                if tokens and isinstance(attentions, np.ndarray) and attentions.size > 0:
                    fig = plot_attention_heatmap(tokens, attentions, title=title)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Could not generate attention visualization: No valid tokens or attention data.")
            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")
    
    # Entity detection
    if st.button("Detect Biomedical Entities"):
        try:
            entities = detect_entities(text)
            
            if entities:
                st.subheader("Detected Entities")
                for entity, entity_type in entities:
                    st.write(f"- **{entity}** ({entity_type})")
            else:
                st.write("No biomedical entities detected.")
        except Exception as e:
            st.error(f"Error detecting entities: {str(e)}")

if __name__ == "__main__":
    main()