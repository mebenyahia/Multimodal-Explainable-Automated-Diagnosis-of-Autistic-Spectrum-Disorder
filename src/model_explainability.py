import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

def explain_model_shap(model, X, model_name):
    explainer = None
    model_type = type(model).__name__

    try:
        if model_type in ["DecisionTreeClassifier", "RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier"]:
            explainer = shap.TreeExplainer(model)
        elif model_type in ["SVC", "MLPClassifier", "LogisticRegression", "GaussianNB", "KNeighborsClassifier", "LinearDiscriminantAnalysis"]:
            explainer = shap.KernelExplainer(model.predict_proba, X)
        else:
            print(f"Model type {model_type} not supported for SHAP explanations.")
            return

        shap_values = explainer.shap_values(X)

        shap.summary_plot(shap_values, X)
        plt.title(f'SHAP Summary Plot for {model_name}')
        plt.show()

        for i in range(min(5, X.shape[0])):  
            shap.waterfall_plot(shap_values[1][i])
            plt.title(f'SHAP Waterfall Plot for {model_name} - Instance {i+1}')
            plt.show()
    except Exception as e:
        print(f"Failed to explain {model_name} using SHAP: {e}")

def explain_all_models(models, X):
    for name, model in models.items():
        print(f"Explaining model: {name}")
        explain_model_shap(model, X, name)
