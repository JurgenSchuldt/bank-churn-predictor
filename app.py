import pandas as pd
import numpy as np
import streamlit as st
import pickle
import plotly.express as px
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.prepare_features import prepare_features
from src.load_data import load_data

st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .block-container { padding-top: 1.5rem; }
    .dashboard-header {
        background-color: #1e3a5f;
        padding: 16px 24px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .dashboard-title { color: white; font-size: 20px; font-weight: 600; margin: 0; }
    .dashboard-sub { color: #8ab4d4; font-size: 13px; margin: 0; }
    .kpi-card {
        background: white;
        border-left: 4px solid #1e3a5f;
        border-radius: 6px;
        padding: 14px 18px;
    }
    .kpi-value { font-size: 24px; font-weight: 600; color: #1e3a5f; margin: 0; }
    .kpi-label { font-size: 12px; color: #888; margin: 0; }
    .risk-high { color: #d32f2f; font-size: 28px; font-weight: 700; }
    .risk-low { color: #1D9E75; font-size: 28px; font-weight: 700; }
    .section-title {
        font-size: 13px; font-weight: 600;
        color: #1e3a5f; margin-bottom: 8px;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/explainer.pkl", "rb") as f:
        explainer = pickle.load(f)
    return model, explainer

@st.cache_data
def load_dataset():
    df = load_data("data/Bank_Customer_Churn_Prediction.csv")
    X, y = prepare_features(df)
    return df, X, y

@st.cache_data
def get_global_shap(_explainer, _X):
    shap_vals = _explainer.shap_values(_X)
    if isinstance(shap_vals, list):
        sv = np.array(shap_vals[1])
    elif hasattr(shap_vals, 'values'):
        sv = np.array(shap_vals.values)
    else:
        sv = np.array(shap_vals)
    while sv.ndim > 2:
        sv = sv[:, :, -1]
    return sv

model, explainer = load_model()
df, X, y = load_dataset()

st.markdown("""
<div class="dashboard-header">
    <p class="dashboard-title">Bank Churn Predictor</p>
    <p class="dashboard-sub">Predicción de abandono de clientes con explicabilidad SHAP</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Predicción individual", "Análisis global"])

with tab1:
    st.markdown("### Introduce los datos del cliente")

    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.slider("Credit Score", 300, 850, 650)
        age = st.slider("Edad", 18, 92, 40)
        tenure = st.slider("Años como cliente", 0, 10, 5)

    with col2:
        balance = st.number_input("Saldo (€)", 0.0, 300000.0, 50000.0, step=1000.0)
        estimated_salary = st.number_input("Salario estimado (€)", 0.0, 200000.0, 80000.0, step=1000.0)
        products_number = st.selectbox("Número de productos", [1, 2, 3, 4])

    with col3:
        country = st.selectbox("País", ["France", "Germany", "Spain"])
        gender = st.selectbox("Género", ["Male", "Female"])
        credit_card = st.selectbox("¿Tiene tarjeta de crédito?", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
        active_member = st.selectbox("¿Miembro activo?", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No")

    if st.button("Predecir", type="primary"):
        country_enc = {"France": 0, "Germany": 1, "Spain": 2}
        gender_enc = {"Male": 1, "Female": 0}

        input_data = pd.DataFrame([{
            'credit_score': credit_score,
            'country': country_enc[country],
            'gender': gender_enc[gender],
            'age': age,
            'tenure': tenure,
            'balance': balance,
            'products_number': products_number,
            'credit_card': credit_card,
            'active_member': active_member,
            'estimated_salary': estimated_salary
        }])

        prob = model.predict_proba(input_data)[0][1]
        pred = model.predict(input_data)[0]

        st.markdown("---")
        col_res1, col_res2 = st.columns(2)

        with col_res1:
            st.markdown('<p class="section-title">Resultado</p>', unsafe_allow_html=True)
            if pred == 1:
                st.markdown('<p class="risk-high">Alto riesgo de churn</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="risk-low">Bajo riesgo de churn</p>', unsafe_allow_html=True)
            st.markdown(f"**Probabilidad de abandono: {prob*100:.1f}%**")

        with col_res2:
            st.markdown('<p class="section-title">¿Por qué?</p>', unsafe_allow_html=True)
            shap_vals = explainer.shap_values(input_data)
            if isinstance(shap_vals, list):
                sv = np.array(shap_vals[1][0])
            elif hasattr(shap_vals, 'values'):
                sv = np.array(shap_vals.values[0])
            else:
                sv = np.array(shap_vals[0])

            while sv.ndim > 1:
                sv = sv[:, -1]

            sv = sv.flatten()

            shap_df = pd.DataFrame({
                'Feature': input_data.columns.tolist(),
                'SHAP': sv
            }).sort_values('SHAP', key=abs, ascending=False).head(5)

            for _, row in shap_df.iterrows():
                direccion = "aumenta" if row['SHAP'] > 0 else "reduce"
                color = "#d32f2f" if row['SHAP'] > 0 else "#1D9E75"
                st.markdown(
                    f"<span style='color:{color}'>{'▲' if row['SHAP']>0 else '▼'} "
                    f"**{row['Feature']}** {direccion} el riesgo</span>",
                    unsafe_allow_html=True
                )

with tab2:
    st.markdown('<p class="section-title">Importancia global de features (SHAP)</p>', unsafe_allow_html=True)

    sv_all = get_global_shap(explainer, X)

    importancia = pd.DataFrame({
        'Feature': X.columns,
        'Importancia SHAP': np.abs(sv_all).mean(axis=0)
    }).sort_values('Importancia SHAP', ascending=True)

    fig = px.bar(
        importancia, x='Importancia SHAP', y='Feature',
        orientation='h', color_discrete_sequence=['#1e3a5f']
    )
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(t=10, b=10, l=10, r=10),
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-title">Estadísticas del modelo</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="kpi-card">
            <p class="kpi-label">Modelo</p>
            <p class="kpi-value" style="font-size:16px">Random Forest</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="kpi-card">
            <p class="kpi-label">AUC-ROC</p>
            <p class="kpi-value">0.846</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="kpi-card">
            <p class="kpi-label">Clientes analizados</p>
            <p class="kpi-value">10,000</p>
        </div>""", unsafe_allow_html=True)
