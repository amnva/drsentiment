from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords
import os


nltk.download("stopwords", quiet=True)

app = FastAPI(title="Doctor Sentiment API",
    description="Predict sentiment of German doctor reviews")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

tfidf_path = os.path.normpath(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
le_path = os.path.normpath(os.path.join(MODEL_DIR, "label_encoder.joblib"))

tfidf = joblib.load(tfidf_path)
le = joblib.load(le_path)

models = {}
for fname in os.listdir(MODEL_DIR):
    if not fname.endswith(".joblib"):
        continue
    if fname in ["tfidf_vectorizer.joblib", "label_encoder.joblib"]:
        continue

    model_name = os.path.splitext(fname)[0]
    full_path = os.path.normpath(os.path.join(MODEL_DIR, fname))
    models[model_name] = joblib.load(full_path)

german_stopwords = set(stopwords.words("german"))

def process_text(text: str) -> str:
    text = re.sub(r"[^\wäöüßÄÖÜ]", " ", text).lower().strip()
    tokens = text.split()
    filtered = [w for w in tokens if w not in german_stopwords]
    return " ".join(filtered)

class PredictionRequest(BaseModel):
    text: str
    model_name: str = "Logistic_Regression"  # default model

class PredictionResponse(BaseModel):
    sentiment: str
    model_used: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Preprocess input text
    processed = process_text(request.text)
    # Vectorize
    X = tfidf.transform([processed])

    chosen = request.model_name
    if chosen not in models:
        chosen = "Logistic_Regression"

    model = models[chosen]
    y_pred = model.predict(X)
    sentiment = le.inverse_transform(y_pred)[0]

    return {"sentiment": sentiment, "model_used": chosen}

@app.get("/models")
async def list_models():
    return {"available_models": list(models.keys())}

