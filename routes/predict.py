from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib
import os

router = APIRouter()

MODEL_PATH = "./models/naive_bayes_model.pkl"
VECTORIZER_PATH = "./models//tfidf_vectorizer.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    raise FileNotFoundError("Model or vectorizer not found. Train the model first.")

class ReviewInput(BaseModel):
    text: str = ""

@router.post("/predict/")
def predict_sentiment(review: ReviewInput):
    if not review.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    text_vectorized = vectorizer.transform([review.text])
    prediction = model.predict(text_vectorized)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    return {"sentiment": sentiment}