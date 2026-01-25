"""BioBERT model wrapper for attention weight extraction."""

import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


class BioBERTAttention:
    """Wrapper class for BioBERT model with attention extraction capabilities."""

    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1") -> None:
        """Initialize the BioBERT model and tokenizer.

        Args:
            model_name: HuggingFace model identifier for BioBERT variant.
        """
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: PreTrainedModel = AutoModel.from_pretrained(
            model_name, output_attentions=True
        )
        self.model.eval()

    def get_attention_weights(
        self, text: str, layer: int = 0, head: int = 0
    ) -> tuple[list[str], npt.NDArray[np.float32]]:
        """Get attention weights for a specific layer and head.

        Args:
            text: Input text to process.
            layer: Transformer layer index (0-11 for base model).
            head: Attention head index (0-11 for base model).

        Returns:
            Tuple of (tokens, attention_weights) where attention_weights
            is a 2D array of shape (seq_len, seq_len).
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tokens: list[str] = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        with torch.no_grad():
            outputs = self.model(**inputs)
            attentions: npt.NDArray[np.float32] = outputs.attentions[layer][0][head].numpy()

        return tokens, attentions

    def get_average_attention(
        self, text: str
    ) -> tuple[list[str], npt.NDArray[np.float32]]:
        """Get average attention across all layers and heads.

        Args:
            text: Input text to process.

        Returns:
            Tuple of (tokens, attention_weights) where attention_weights
            is averaged across all layers and heads.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tokens: list[str] = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        with torch.no_grad():
            outputs = self.model(**inputs)
            attentions: npt.NDArray[np.float32] = (
                torch.stack(outputs.attentions).mean(dim=0).mean(dim=1)[0].numpy()
            )

        return tokens, attentions