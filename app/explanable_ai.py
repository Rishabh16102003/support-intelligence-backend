import torch
import numpy as np
import pandas as pd

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

def attention(subject, description):

    text = subject + " " + description

    text_dl = clean_text_dl(text)

    encoding = queue_tokenizer(
        text_dl,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = queue_model(
            **encoding,
            output_attentions=True
        )

        queue_pred_encoded = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

    queue_pred = queue_encoder.inverse_transform(
        [queue_pred_encoded]
    )[0]

    # -------------------------
    # Attention Extraction
    # -------------------------

    attentions = outputs.attentions

    last_layer = attentions[-1]

    # Average across heads
    avg_attention = last_layer[0].mean(dim=0)

    # CLS -> token attention
    attention_scores = avg_attention[0]

    tokens = queue_tokenizer.convert_ids_to_tokens(
        encoding["input_ids"][0]
    )

    attention_df = pd.DataFrame({
        "token": tokens,
        "attention": attention_scores.cpu().numpy()
    })

    attention_df = attention_df.sort_values(
        by="attention",
        ascending=False
    )

    attention_df_exclusive = attention_df[
        ~attention_df["token"].isin(
            ["[CLS]", "[SEP]", "[PAD]", ".", ","]
        )
    ]

    attention_df_exclusive = attention_df_exclusive[
        ~attention_df_exclusive["token"].str.startswith("##")
    ]

    attention_df_exclusive = attention_df_exclusive.head(15)

    return (
        queue_pred,
        attention_df.head(20),
        attention_df_exclusive
    )

# def attention(subject,description):
     
#     text = subject + " " + description
#     text_dl = clean_text_dl(text)

#     encoding = queue_tokenizer(
#         text_dl,
#         truncation=True,
#         padding=True,
#         max_length=128,
#         return_tensors="pt"
#     )
#     with torch.no_grad():

#         outputs = queue_model(**encoding)

#         queue_pred_encoded = torch.argmax(
#             outputs.logits,
#             dim=1
#         ).item()


#     queue_pred = queue_encoder.inverse_transform(
#         [queue_pred_encoded]
#     )[0]
        
#     # extracting last layer attention
#     attentions = outputs.attentions

#     last_layer = attentions[-1]

#     attention_scores = (
#     last_layer[0]
#     .mean(dim=0)
#     .mean(dim=0)
#     )

#     tokens = queue_tokenizer.convert_ids_to_tokens(
#     encoding["input_ids"][0]
#     )

#     attention_df = pd.DataFrame({
#     "token": tokens,
#     "attention": attention_scores.cpu().numpy()
#     })

#     attention_df = attention_df.sort_values(
#     "attention",
#     ascending=False
#     )

#     attention_df=attention_df.head(20)


#     attention_df_exclusive = attention_df[
#     ~attention_df["token"].isin(
#         ["[CLS]", "[SEP]", "[PAD]",',','.']
#     )
#     ]

#     attention_df_exclusive = attention_df_exclusive[
#     ~attention_df_exclusive["token"].str.startswith("##")
#     ]
#     attention_df_exclusive=attention_df_exclusive.head(15)

#     return queue_pred,attention_df,attention_df_exclusive

# def shap_explain(subject,description):
#     pass