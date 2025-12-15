import joblib
from data.load_data import load_data
from data.preprocess import preprocess_text
from features.vectorize import vectorize_text
from models.model_definitions import MODELS
from models.train import train_models
from models.evaluate import evaluate_model
from visualization.plot_utils import plot_class_distribution, plot_model_performance

def main():
    # Data pipeline
    df = load_data(sample_frac=0.3)
    plot_class_distribution(df)
    df = preprocess_text(df)
    
    # Feature engineering
    X, y, tfidf, le = vectorize_text(df)
    
    # Model training
    X_train, X_test, y_train, y_test = train_models(X, y)
    
    # Training and evaluation
    results = []
    for name, model in MODELS.items():
        print(f"\n{'='*40}\nTraining {name}...\n{'='*40}")
        model.fit(X_train, y_train)
        joblib.dump(model, f'saved_models/{name}.joblib')
        
        metrics = evaluate_model(model, name, X_test, y_test, le)
        results.append(metrics)
    
    # Print and plot results
    results_df = pd.DataFrame(results).set_index('Model')
    plot_model_performance(results_df)
    print("\nFinal Performance:")
    print(results_df.sort_values(by='F1', ascending=False))

if __name__ == "__main__":
    main()
