import os
import torch
import torch.nn as nn
import streamlit as st

from huggingface_hub import hf_hub_download

from transformers import (
    DistilBertModel,
    DistilBertTokenizerFast
)

# =====================================================================
# LABELS & THRESHOLDS ALIGNED WITH THESIS CHAPTER 4 VALIDATION MAXIMA
# =====================================================================
PILLAR_MAP = {
    0: "Assessment",
    1: "Course Structure",
    2: "Lecturer",
    3: "Other",
    4: "Subject Material"
}

PILLAR_THRESHOLDS = [
    0.15,  # Assessment
    0.40,  # Course Structure
    0.60,  # Lecturer
    0.45,  # Other
    0.20   # Subject Material
]

SUGGESTION_THRESHOLD = 0.40

SENTIMENT_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


MODEL_PATH = "dep/model/best_distilbert_model.pt"
MODEL_REPO_ID = "mareeha11/student-feedback-distilbert"
MODEL_FILENAME = "best_distilbert_model.pt"


def resolve_model_path():
    # Keep using the local checkpoint when it exists.
    if os.path.isfile(MODEL_PATH):
        return MODEL_PATH

    # Otherwise, retrieve the private checkpoint from Hugging Face.
    token = os.getenv("HF_TOKEN")

    if not token:
        try:
            token = st.secrets.get("HF_TOKEN")
        except Exception:
            token = None

    if not token:
        raise RuntimeError(
            "The trained model is unavailable locally. "
            "Configure HF_TOKEN in the app's deployment secrets."
        )

    return hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        token=token,
    )

# =====================================================================
# DEEP NEURAL NETWORK ARCHITECTURE: SHARED ENCODER MULTI-TASK LEARNING
# =====================================================================
class DistilBERTMultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.distilbert_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden_size = 768
        self.dropout_layer = nn.Dropout(0.3)
        self.pillar_head = nn.Linear(hidden_size, 5)
        self.sentiment_head = nn.Linear(hidden_size, 3)
        self.suggestion_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        encoder_output = self.distilbert_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = encoder_output.last_hidden_state[:, 0]
        cls_embedding = self.dropout_layer(cls_embedding)
        
        pillar_logits = self.pillar_head(cls_embedding)
        sentiment_logits = self.sentiment_head(cls_embedding)
        suggestion_logits = self.suggestion_head(cls_embedding)
        
        return pillar_logits, sentiment_logits, suggestion_logits

# =====================================================================
# DECOUPLED CORE MODEL CACHING LAYER (FIXES CIRCULAR IMPORT ERROR)
# =====================================================================

@st.cache_resource
def initialize_deep_learning_backbone():
    """Loads and caches the trained multi-task model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBERTMultiTaskModel()

    state_dict = torch.load(
        resolve_model_path(),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased"
    )
    return model, tokenizer, device


# =====================================================================
# INFERENCE EXECUTION PIPELINE FUNCTION
# =====================================================================
def predict(text):
    """Executes evaluation string ingestion using pre-cached model states."""
    # Pull model, tokenizer, device instantly from local cached resource pool
    model, tokenizer, device = initialize_deep_learning_backbone()

    inputs = tokenizer(
        str(text),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        pillar_logits, sentiment_logits, suggestion_logits = model(
            inputs["input_ids"],
            inputs["attention_mask"]
        )

    # 1. PILLAR EXTRACTION (MULTI-LABEL)
    pillar_probs = torch.sigmoid(pillar_logits).squeeze().cpu().numpy()
    pillars = []
    pillar_scores_map = {}

    for idx, prob in enumerate(pillar_probs):
        if prob >= PILLAR_THRESHOLDS[idx]:
            p_name = PILLAR_MAP[idx]
            pillars.append(p_name)
            pillar_scores_map[p_name] = round(float(prob) * 100, 2)

    # 2. SENTIMENT EXTRACTION (MULTI-CLASS)
    sentiment_probabilities = torch.softmax(sentiment_logits, dim=1).squeeze()
    sentiment_id = int(torch.argmax(sentiment_probabilities).item())
    sentiment = SENTIMENT_MAP[sentiment_id]
    sentiment_conf = round(float(sentiment_probabilities[sentiment_id]) * 100, 2)

    # 3. SUGGESTION EXTRACTION (BINARY)
    suggestion_probability = float(torch.sigmoid(suggestion_logits).item())
    suggestion = "Yes" if suggestion_probability >= SUGGESTION_THRESHOLD else "No"
    suggestion_conf = round(suggestion_probability * 100, 2)

    return {
        "pillars": pillars,
        "pillar_scores": pillar_scores_map if pillars else {},
        "sentiment": sentiment,
        "sentiment_confidence": sentiment_conf,
        "suggestion": suggestion,
        "suggestion_confidence": suggestion_conf
    }