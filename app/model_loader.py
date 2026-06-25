import joblib

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)


# =========================
# LOAD QUEUE MODEL
# =========================

queue_model = DistilBertForSequenceClassification.from_pretrained(
    "models/queue_model",
    output_attentions=True,

)

queue_tokenizer = DistilBertTokenizerFast.from_pretrained(
    "models/queue_tokenizer",
     output_attentions=True
)


# =========================
# LOAD LABEL ENCODERS
# =========================

queue_encoder = joblib.load(
    "models/queue_label_encoder.pkl"
)

type_encoder = joblib.load(
    "models/type_label_encoder.pkl"
)

priority_encoder = joblib.load(
    "models/priority_label_encoder.pkl"
)


# =========================
# LOAD TFIDF
# =========================

tfidf = joblib.load(
    "models/tfidf_type.pkl"
)


# =========================
# LOAD XGBOOST MODELS
# =========================

type_model = joblib.load(
    "models/type_xgb.pkl"
)

priority_model = joblib.load(
    "models/priority_xgb.pkl"
)