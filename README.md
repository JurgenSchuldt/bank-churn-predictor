# Bank Churn Predictor

Modelo de machine learning para predecir el abandono de clientes bancarios, con explicabilidad mediante SHAP values.

Live demo: https://bank-churn-prediictor.streamlit.app/

## Qué hace

Predice la probabilidad de que un cliente abandone el banco basándose en su perfil — edad, saldo, productos contratados, actividad y otros factores. Incluye explicabilidad completa: para cada predicción el modelo muestra qué variables aumentan o reducen el riesgo y en qué medida.

## Funcionalidades

- Predicción individual con sliders y selectores interactivos
- Explicación SHAP por cliente — qué factores aumentan o reducen el riesgo
- Análisis global de importancia de features
- Comparación de tres modelos: Regresión Logística, Random Forest y XGBoost

## Resultados del modelo

- Mejor modelo: Random Forest
- AUC-ROC: 0.846
- F1-score (churn): 0.58
- Dataset: 10,000 clientes bancarios

## Tecnologías

- Python 3.12
- scikit-learn
- XGBoost
- SHAP
- Streamlit
- pandas

## Cómo ejecutarlo
```bash
git clone https://github.com/JurgenSchuldt/bank-churn-predictor.git
cd bank-churn-predictor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py