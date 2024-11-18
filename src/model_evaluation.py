import json
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import evaluate
from evaluate import visualization

accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")
roc_auc_metric = evaluate.load("roc_auc")
confusion_matrix_metric = evaluate.load("confusion_matrix")

def evaluate_model(y_true, y_pred, y_prob=None):
    print("Evaluating model...")
    print("y_true (type):", type(y_true))
    print("y_pred (type):", type(y_pred))
    print("y_true (sample):", y_true[:5])  
    print("y_pred (sample):", y_pred[:5])  
    if y_prob is not None:
        print("y_prob (type):", type(y_prob))
        print("y_prob (sample):", y_prob[:5]) 

    y_true = y_true.tolist() if isinstance(y_true, pd.Series) else y_true
    y_pred = y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred

    metrics = {
        'Accuracy': accuracy_metric.compute(predictions=y_pred, references=y_true)['accuracy'],
        'Precision': precision_metric.compute(predictions=y_pred, references=y_true, average='weighted')['precision'],
        'Recall': recall_metric.compute(predictions=y_pred, references=y_true, average='weighted')['recall'],
        'F1-Score': f1_metric.compute(predictions=y_pred, references=y_true, average='weighted')['f1'],
        'Sensitivity (ASD)': recall_metric.compute(predictions=y_pred, references=y_true, pos_label=1)['recall'],
        'Specificity (non-ASD)': recall_metric.compute(predictions=y_pred, references=y_true, pos_label=0)['recall'],
        'ROC AUC': roc_auc_metric.compute(prediction_scores=y_prob[:, 1], references=y_true)['roc_auc'] if y_prob is not None else 'N/A',
        'Confusion Matrix': confusion_matrix_metric.compute(predictions=y_pred, references=y_true)['confusion_matrix'].tolist()
    }
    return metrics

def cross_validate_model(model, X, y, cv=10):
    print(f"Cross-validating model {model}...")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    y_pred = cross_val_predict(model, X, y, cv=skf)
    if hasattr(model, "predict_proba"):
        y_prob = cross_val_predict(model, X, y, cv=skf, method='predict_proba')
    else:
        y_prob = cross_val_predict(model, X, y, cv=skf, method='decision_function')
    
    return evaluate_model(y, y_pred, y_prob)

def evaluate_all_models(models, X, y, cv=10):
    print("Evaluating all models...")
    evaluation_results = {}
    for name, model in models.items():
        print(f"Evaluating model: {name}")
        metrics = cross_validate_model(model, X, y, cv=cv)
        evaluation_results[name] = metrics
    return evaluation_results

def save_evaluation_results(evaluation_results, file_path):
    with open(file_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"Results saved to {file_path}")

def visualize_evaluation_results(evaluation_results):
    data = [{metric: value for metric, value in metrics.items() if metric != 'Confusion Matrix'}
            for metrics in evaluation_results.values()]
    model_names = list(evaluation_results.keys())
    
    fig = visualization.radar_plot(
        data=data,
        names=model_names,
        config={
            "rad_ln_args": {"visible": True},
            "outer_ring": {"visible": True},
            "angle_ln_args": {"visible": True},
            "rgrid_tick_lbls_args": {"fontsize": 12},
            "theta_tick_lbls": {"fontsize": 12},
            "theta_tick_lbls_pad": 3,
            "theta_tick_lbls_brk_lng_wrds": True,
            "theta_tick_lbls_txt_wrap": 15,
            "incl_endpoint": False,
            "marker": "o",
            "markersize": 3,
            "legend_loc": "upper right",
            "bbox_to_anchor": (1.1, 1.05)
        }
    )
    fig.show()

if __name__ == "__main__":
    import joblib

    data = pd.read_csv('../data/processed_data.csv')
    X = data.drop(columns='DX_GROUP')
    y = data['DX_GROUP'] - 1  

    trained_models = {
        'Decision Tree': joblib.load('../models/decision_tree_model.pkl'),
        'Random Forest': joblib.load('../models/random_forest_model.pkl'),
        'SVC': joblib.load('../models/svc_model.pkl'),
        'MLP': joblib.load('../models/mlp_model.pkl'),
        'LDA': joblib.load('../models/lda_model.pkl'),
        'KNN': joblib.load('../models/knn_model.pkl'),
        'Gradient Boosting': joblib.load('../models/gradient_boosting_model.pkl'),
        'XGBoost': joblib.load('../models/xgboost_model.pkl'),
        'Logistic Regression': joblib.load('../models/logistic_regression_model.pkl'),
        'Naive Bayes': joblib.load('../models/naive_bayes_model.pkl')
    }

    evaluation_results = evaluate_all_models(trained_models, X, y, cv=10)

    save_evaluation_results(evaluation_results, '../results/evaluation_metrics.json')
    
    visualize_evaluation_results(evaluation_results)

    print("Model evaluation complete. Results saved to '../results/evaluation_metrics.json'.")
