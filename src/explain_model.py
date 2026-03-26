import pandas as pd
import numpy as np
import shap
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_data import load_data
from prepare_features import prepare_features

def explain_model():
    df = load_data("data/Bank_Customer_Churn_Prediction.csv")
    X, y = prepare_features(df)

    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)

    print("Calculando SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Compatibilidad con distintas versiones de shap
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif hasattr(shap_values, 'values'):
        sv = shap_values.values[:, :, 1] if shap_values.values.ndim == 3 else shap_values.values
    else:
        sv = shap_values

    sv = np.array(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    print("\n=== IMPORTANCIA DE FEATURES (SHAP) ===")
    importancia = pd.DataFrame({
        'feature': X.columns,
        'shap_mean': np.abs(sv).mean(axis=0)
    }).sort_values('shap_mean', ascending=False)
    print(importancia.to_string(index=False))

    os.makedirs("models", exist_ok=True)
    with open("models/explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)

    print("\nExplainer guardado en models/explainer.pkl")
    return explainer, sv, X

if __name__ == "__main__":
    explain_model()