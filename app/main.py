from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import TicketRequest
from app.prediction import predict_ticket
from app.explanable_ai import attention

app = FastAPI(title="Customer Support AI API")

# Allow Streamlit frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict this later to your Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Root endpoint
# -------------------------
@app.get("/")
def home():
    return {"message": "Customer Support AI API Running"}

# -------------------------
# Health endpoint
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
def predict(data: TicketRequest):
    result = predict_ticket(
        data.subject,
        data.description
    )
    return result

# -------------------------
# Attention endpoint
# -------------------------
@app.post("/attention")
def predict_attention(data: TicketRequest):
    pred, df, df_exclusive = attention(
        data.subject,
        data.description
    )

    return {
        "pred": pred,
        "df": df.to_dict(orient="records"),
        "df_exclusive": df_exclusive.to_dict(orient="records")
    }