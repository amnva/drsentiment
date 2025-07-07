
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY saved_models/ /saved_models/ 
COPY app/     app/

RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"

CMD ["uvicorn", "app.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
