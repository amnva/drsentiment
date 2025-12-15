from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

MODELS = {
    "Logistic_Regression": LogisticRegression(max_iter=500, tol=1e-3, solver='saga', n_jobs=-1),
    "Random_Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
    "Linear_SVM": LinearSVC(class_weight='balanced', dual=False, tol=1e-3),
    "Multinomial_NB": MultinomialNB(),
    "XGBoost": XGBClassifier(tree_method='hist', n_jobs=-1),
    "LightGBM": LGBMClassifier(n_jobs=-1),
    "MLP": MLPClassifier(hidden_layer_sizes=(64,32), early_stopping=True,
                         batch_size=512, max_iter=150, solver='adam',
                         n_iter_no_change=5, learning_rate_init=0.001, tol=1e-3),
    "Extra_Trees": ExtraTreesClassifier(class_weight='balanced', n_estimators=50,
                                       max_depth=15, bootstrap=True, n_jobs=-1, max_samples=0.8),
    "AdaBoost": AdaBoostClassifier(n_estimators=40, learning_rate=1.0,
                                  estimator=DecisionTreeClassifier(max_depth=1)),
    "K_Neighbors": KNeighborsClassifier(n_neighbors=3, algorithm='brute', metric='cosine')
}
