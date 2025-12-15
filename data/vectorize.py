from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def vectorize_text(df, max_features=1000):
    tfidf = TfidfVectorizer(max_features=max_features)
    le = LabelEncoder()
    
    X = tfidf.fit_transform(df['processed_comment'])
    y = le.fit_transform(df['sentiment'])
    
    os.makedirs('saved_models', exist_ok=True)
    joblib.dump(tfidf, 'saved_models/tfidf_vectorizer.joblib')
    joblib.dump(le, 'saved_models/label_encoder.joblib')
    
    return X, y, tfidf, le
