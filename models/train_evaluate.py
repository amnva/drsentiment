import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                            ConfusionMatrixDisplay, f1_score)
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import joblib


def train_models(X, y, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0, stratify=y
    )
    
    ros = RandomOverSampler(random_state=0)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    
    return X_train, X_test, y_train, y_test


def evaluate_model(model, name, X_test, y_test, le):
    y_pred = model.predict(X_test)
    
    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred, average='weighted')
    }
    
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=le.classes_, cmap='Blues'
    )
    plt.title(f'Confusion Matrix - {name}')
    plt.show()
    
    return metrics
