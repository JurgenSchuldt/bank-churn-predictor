import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def explore_data(df):
    print("=== DIMENSIONES ===")
    print(f"Filas: {df.shape[0]:,} | Columnas: {df.shape[1]}")

    print("\n=== COLUMNAS ===")
    print(df.dtypes)

    print("\n=== PRIMERAS 5 FILAS ===")
    print(df.head())

    print("\n=== VALORES NULOS ===")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0] if nulls.sum() > 0 else "Sin valores nulos")

    print("\n=== DISTRIBUCIÓN DE CHURN ===")
    print(df['churn'].value_counts())
    print(f"% Churn: {df['churn'].mean()*100:.1f}%")

    print("\n=== ESTADÍSTICAS BÁSICAS ===")
    print(df.describe())

if __name__ == "__main__":
    df = load_data("data/Bank_Customer_Churn_Prediction.csv")
    explore_data(df)