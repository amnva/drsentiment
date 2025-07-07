# German Doctor Review Sentiment Classifier

A machine learning-powered classifier for detecting sentiment in German-language doctor reviews. Includes model training, evaluation, and a FastAPI-based REST API for real-time inference.

## Features


- ⚡ Supports multiple ML models (SVM, Logistic Regression, LightGBM, etc.)
- 🧱 Modular architecture for preprocessing, vectorization, and training
- 🌐 Real-time prediction via FastAPI (`/predict`)
- 🐳 Dockerized for portable deployment
- 📊 Includes tools for plotting and evaluation (confusion matrix, scores, etc.)

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/amnva/drsentiment.git
cd drsentiment
```

### 2. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### 3. Run the training pipeline

```bash
python main.py
```

### 4. Run the API Locally

```bash
uvicorn app.fastapi_app:app --reload
```

### 5. Build and Run with Docker

```bash
docker build -t drsentiment .
docker run -p 8000:8000 drsentiment
```

### Example API Usage
Access via browser:
```bash
http://localhost:8000/docs
```
Example input:

```bash
POST /predict
{
  "text": "Gut und aufschlussreich. Behandlung erfolgreich",
  "model_name": "Logistic_Regression"
}

```

Via Terminal (cURL)

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"text": "Gut und aufschlussreich. Behandlung erfolgreich","model_name": "Logistic_Regression"}'

```
