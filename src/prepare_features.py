import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_features(df):
    data = df.copy()
    
    # Eliminar columnas que no aportan al modelo
    data = data.drop(columns=['customer_id'])
    
    # Convertir variables categóricas a numéricas
    le = LabelEncoder()
    data['country'] = le.fit_transform(data['country'])
    data['gender'] = le.fit_transform(data['gender'])
    
    # Separar features y variable objetivo
    X = data.drop(columns=['churn'])
    y = data['churn']
    
    print("=== FEATURES DEL MODELO ===")
    print(list(X.columns))
    print(f"\nDistribución churn — 0: {(y==0).sum():,} | 1: {(y==1).sum():,}")
    
    return X, y

if __name__ == "__main__":
    from load_data import load_data
    df = load_data("data/Bank_Customer_Churn_Prediction.csv")
    X, y = prepare_features(df)
