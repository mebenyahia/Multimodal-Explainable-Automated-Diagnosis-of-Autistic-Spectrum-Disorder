import pandas as pd
import numpy as np
import json
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def perform_grid_search(X_train, y_train):
    param_grids = {
        'Random Forest': {
            'classifier__n_estimators': [50, 100, 200, 500],
            'classifier__max_depth': [3, 5, 10, 20, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__bootstrap': [True, False],
            'classifier__max_features': ['sqrt', 'log2', None]
        },
        'Logistic Regression': [
            {
                'classifier__penalty': ['l1', 'l2'],
                'classifier__C': [0.01, 0.1, 1, 10, 100],
                'classifier__solver': ['saga'],
                'classifier__tol': [1e-4, 1e-3, 1e-2],
                'classifier__max_iter': [100, 200, 500, 1000, 1500, 2000, 2500, 3000]
            },
            {
                'classifier__penalty': ['elasticnet'],
                'classifier__C': [0.01, 0.1, 1, 10, 100],
                'classifier__solver': ['saga'],
                'classifier__l1_ratio': [0.1, 0.15, 0.5, 0.85, 0.9],
                'classifier__tol': [1e-4, 1e-3, 1e-2],
                'classifier__max_iter': [100, 200, 500, 1000, 1500, 2000, 2500, 3000]
            }
        ],
        'Naive Bayes': {
            'classifier__var_smoothing': [1e-09, 1e-08, 1e-07, 1e-06]
        },
        'LDA': {
            'classifier__solver': ['lsqr', 'eigen'],
            'classifier__shrinkage': [None, 'auto'] + list(np.linspace(0.0, 1.0, 10))
        },
        'KNN': {
            'classifier__n_neighbors': [3, 5, 11, 19, 21],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'Gradient Boosting': {
            'classifier__n_estimators': [50, 100, 200, 500],
            'classifier__learning_rate': [0.01, 0.1, 0.2, 0.3],
            'classifier__max_depth': [3, 5, 10, 20],
            'classifier__subsample': [0.6, 0.8, 1.0]
        },
        'XGBoost': {
            'classifier__n_estimators': [50, 100, 200, 500],
            'classifier__learning_rate': [0.01, 0.1, 0.2, 0.3],
            'classifier__max_depth': [3, 5, 10, 20],
            'classifier__subsample': [0.6, 0.8, 1.0],
            'classifier__colsample_bytree': [0.6, 0.8, 1.0]
        }
    }

    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
        'Naive Bayes': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis(),
        'KNN': KNeighborsClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    best_estimators = {}
    evaluation_results = {}
    for model_name, model in models.items():
        print(f"Performing Grid Search for {model_name}...")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        param_grid = param_grids[model_name]
        grid_search = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(10), scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_estimators[model_name] = grid_search.best_estimator_

        evaluation_results[model_name] = {
            'best_params': grid_search.best_params_,
            'best_cross_val_accuracy': grid_search.best_score_
        }

        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best cross-validation accuracy for {model_name}: {grid_search.best_score_}")

    return best_estimators, evaluation_results

def evaluate_on_test_set(best_estimators, X_test, y_test, evaluation_results):
    for model_name, estimator in best_estimators.items():
        test_score = estimator.score(X_test, y_test)
        evaluation_results[model_name]['test_accuracy'] = test_score
        print(f"Test accuracy for {model_name}: {test_score}")

def save_tuning_results(evaluation_results, file_path):
    with open(file_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"Hyperparameter tuning and evaluation results saved to '{file_path}'")

if __name__ == "__main__":
    data = pd.read_csv('../data/processed_data.csv')
    X = data.drop(columns='DX_GROUP')
    y = data['DX_GROUP'] - 1  

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    best_estimators, evaluation_results = perform_grid_search(X_train, y_train)
    evaluate_on_test_set(best_estimators, X_test, y_test, evaluation_results)
    save_tuning_results(evaluation_results, '../results/hyperparameter_tuning_results.json')
