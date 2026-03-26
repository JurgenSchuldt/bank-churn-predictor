import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_data import load_data
from prepare_features import prepare_features

def train_and_evaluate():
    # Cargar y preparar datos
    df = load_data("data/Bank_Customer_Churn_Prediction.csv")
    X, y = prepare_features(df)

    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} filas | Test: {len(X_test):,} filas\n")

    modelos = {
        "Regresión Logística": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }

    resultados = {}

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = report['1']['f1-score']

        resultados[nombre] = {'modelo': modelo, 'auc': auc, 'f1': f1}

        print(f"=== {nombre} ===")
        print(f"AUC-ROC: {auc:.4f} | F1 (churn): {f1:.4f}")
        print(classification_report(y_test, y_pred))

    # Guardar el mejor modelo
    mejor = max(resultados, key=lambda x: resultados[x]['f1'])
    print(f"Mejor modelo: {mejor}")

    os.makedirs("models", exist_ok=True)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(resultados[mejor]['modelo'], f)

    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)

    print("Modelo guardado en models/best_model.pkl")

    return resultados[mejor]['modelo'], X_test, y_test, X.columns.tolist()

if __name__ == "__main__":
    train_and_evaluate()

