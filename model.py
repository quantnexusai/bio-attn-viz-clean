import torch
from transformers import AutoModel, AutoTokenizer

class BioBERTAttention:
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.model.eval()

    def get_attention_weights(self, text, layer=0, head=0):
        """Get attention weights for a specific layer and head"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        with torch.no_grad():
            outputs = self.model(**inputs)
            attentions = outputs.attentions[layer][0][head].numpy()

        return tokens, attentions

    def get_average_attention(self, text):
        """Get average attention across all layers and heads"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        with torch.no_grad():
            outputs = self.model(**inputs)
            attentions = torch.stack(outputs.attentions).mean(dim=0).mean(dim=1)[0].numpy()

        return tokens, attentions