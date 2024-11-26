import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

def perform_rfe(file_path, target_column, n_features_to_select, features_to_drop):
    """
    RFE with standard scaling and cross-validation to ensure no data leakage.

    -> target_column (str): DX_group.
    -> features_to_drop (list) to apply rfe only on fmri and drop vip, piq ect.
    """

    data = pd.read_csv(file_path)

    labels = data[target_column]
    features = data.drop(columns=[target_column] + features_to_drop)

    base_model = LogisticRegression(max_iter=1000)

    rfe = RFE(estimator=base_model, n_features_to_select=n_features_to_select, step=1)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),       
        ('feature_selection', rfe),         
        ('classification', base_model)     
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, features, labels, cv=cv, scoring='accuracy')

    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', rfe)
    ])
    final_pipeline.fit(features, labels)
    selected_features = features.columns[rfe.support_].tolist()

    mean_cv_accuracy = np.mean(cv_scores)
    return selected_features, mean_cv_accuracy
