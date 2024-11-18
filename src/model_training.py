from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model

def train_svc(X_train, y_train):
    model = SVC(probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_mlp(X_train, y_train):
    model = MLPClassifier(early_stopping=True,  max_iter=1000000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_lda(X_train, y_train):
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def train_all_models(X_train, y_train):
    models = {
        'Decision Tree': train_decision_tree(X_train, y_train),
        'Random Forest': train_random_forest(X_train, y_train),
        'SVC': train_svc(X_train, y_train),
        'MLP': train_mlp(X_train, y_train),
        'LDA': train_lda(X_train, y_train),
        'KNN': train_knn(X_train, y_train),
        'Gradient Boosting': train_gradient_boosting(X_train, y_train),
        'XGBoost': train_xgboost(X_train, y_train),
        'Logistic Regression': train_logistic_regression(X_train, y_train),
        'Naive Bayes': train_naive_bayes(X_train, y_train),
    }
    return models

def save_all_models(models, directory='models'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for name, model in models.items():
        joblib.dump(model, os.path.join(directory, f'{name.lower().replace(" ", "_")}_model.pkl'))
    print("All models saved to the 'models' directory.")

if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data = pd.read_csv('../data/processed_data.csv')
    X = data.drop(columns='DX_GROUP')
    y = data['DX_GROUP'] - 1  
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    trained_models = train_all_models(X_train, y_train)

    save_all_models(trained_models)
    
    print("Model training complete.")
