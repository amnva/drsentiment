import matplotlib.pyplot as plt

def plot_class_distribution(df):
    plt.figure(figsize=(8,5))
    df['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Class Distribution')
    plt.show()

def plot_model_performance(results_df):
    plt.figure(figsize=(10, 6))
    results_df.sort_values(by='F1', ascending=False).plot(
        y=['Accuracy', 'F1'], kind='bar', title='Model Performance'
    )
    plt.show()
