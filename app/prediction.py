import torch
import numpy as np

from scipy.sparse import hstack

from src.text_preprocess import clean_text_dl,clean_text_ml

from app.model_loader import (
    queue_model,
    queue_tokenizer,
    queue_encoder,
    tfidf,
    type_model,
    type_encoder,
    priority_model,
    priority_encoder
)


def predict_ticket(subject, description):

    # =========================
    # COMBINE TEXT
    # =========================

    text = subject + " " + description


    # =========================
    # CLEAN TEXT
    # =========================

    text_dl = clean_text_dl(text)


    # =========================
    # QUEUE PREDICTION
    # =========================

    encoding = queue_tokenizer(
        text_dl,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )


    with torch.no_grad():

        outputs = queue_model(**encoding)

        queue_pred_encoded = torch.argmax(
            outputs.logits,
            dim=1
        ).item()


    queue_pred = queue_encoder.inverse_transform(
        [queue_pred_encoded]
    )[0]


    # =========================
    # TFIDF FEATURES
    # =========================
    text_ml = clean_text_ml(text)
    tfidf_features = tfidf.transform([text_ml])


    # =========================
    # TYPE PREDICTION
    # =========================

    type_pred_encoded = type_model.predict(
        tfidf_features
    )[0]


    type_pred = type_encoder.inverse_transform(
        [type_pred_encoded]
    )[0]


    # =========================
    # PRIORITY PREDICTION
    # =========================

    extra_features = np.array([
        [queue_pred_encoded, type_pred_encoded]
    ])


    final_features = hstack([
        tfidf_features,
        extra_features
    ])


    priority_pred_encoded = priority_model.predict(
        final_features
    )[0]


    priority_pred = priority_encoder.inverse_transform(
        [priority_pred_encoded]
    )[0]


    # =========================
    # RETURN RESULTS
    # =========================

    return {

        "queue_prediction": queue_pred,

        "type_prediction": type_pred,

        "priority_prediction": priority_pred
    }