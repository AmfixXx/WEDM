import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    # 1. Wczytanie danych z pliku kwantyli: kwantyle3.csv
    data_path = "kwantyle3.csv"
    df = pd.read_csv(data_path, sep=';')

    # Spodziewamy się, że df ma kolumny: [H, E, Q, speed_mm_min]
    df.dropna(subset=['H','E','Q','speed_mm_min'], inplace=True)

    # 2. Rozdzielenie na cztery podzbiory: Q=0.15, 0.35, 0.5, 0.6
    df_015 = df[df['Q'] == 0.15].copy()
    df_035 = df[df['Q'] == 0.35].copy()
    df_05  = df[df['Q'] == 0.5 ].copy()
    df_06  = df[df['Q'] == 0.6 ].copy()

    # 3. Łączymy i robimy one-hot encoding kolumny E (E1..E5, a resztę usuwamy)
    df_all = pd.concat([df_015, df_035, df_05, df_06], ignore_index=True)
    df_all = pd.get_dummies(df_all, columns=['E'], prefix='E')

    # Zatrzymujemy tylko E_E1..E_E5
    keep_es = ['E_E1','E_E2','E_E3','E_E4','E_E5']
    for col in keep_es:
        if col not in df_all.columns:
            df_all[col] = 0

    e_cols_to_drop = [c for c in df_all.columns if c.startswith('E_') and c not in keep_es]
    df_all.drop(columns=e_cols_to_drop, inplace=True)

    # 4. Teraz df_all ma kolumny: [H, Q, speed_mm_min, E_E1..E_E5]
    #    Rozdzielamy z powrotem
    df_015_enc = df_all[df_all['Q'] == 0.15].copy()
    df_035_enc = df_all[df_all['Q'] == 0.35].copy()
    df_05_enc  = df_all[df_all['Q'] == 0.5 ].copy()
    df_06_enc  = df_all[df_all['Q'] == 0.6 ].copy()

    # Funkcja pomocnicza
    def prepare_xy(df_enc):
        feature_cols = [col for col in df_enc.columns if col not in ('Q','speed_mm_min')]
        X = df_enc[feature_cols].copy()
        y = df_enc['speed_mm_min'].values
        return X, y

    X_015, y_015 = prepare_xy(df_015_enc)
    X_035, y_035 = prepare_xy(df_035_enc)
    X_05,  y_05  = prepare_xy(df_05_enc)
    X_06,  y_06  = prepare_xy(df_06_enc)

    # 5. Tworzymy jedną ramkę do dopasowania scalera:
    X_all_enc = df_all[[c for c in df_all.columns if c not in ('Q','speed_mm_min')]]

    scaler = StandardScaler()
    scaler.fit(X_all_enc)

    X_015_scaled = scaler.transform(X_015)
    X_035_scaled = scaler.transform(X_035)
    X_05_scaled  = scaler.transform(X_05)
    X_06_scaled  = scaler.transform(X_06)

    # 6. Trenujemy 4 osobne modele (GradientBoostingRegressor)
    models = {}

    model_015 = GradientBoostingRegressor(
        n_estimators=150, max_depth=5, random_state=42
    )
    model_015.fit(X_015_scaled, y_015)
    models[0.15] = model_015

    model_035 = GradientBoostingRegressor(
        n_estimators=150, max_depth=5, random_state=42
    )
    model_035.fit(X_035_scaled, y_035)
    models[0.35] = model_035

    model_05 = GradientBoostingRegressor(
        n_estimators=150, max_depth=5, random_state=42
    )
    model_05.fit(X_05_scaled, y_05)
    models[0.5] = model_05

    model_06 = GradientBoostingRegressor(
        n_estimators=150, max_depth=5, random_state=42
    )
    model_06.fit(X_06_scaled, y_06)
    models[0.6] = model_06

    # 7. Zapisujemy modele i scaler
    joblib.dump(models, "models.pkl")

    # Zapisujemy też feature_names
    feature_names = X_015.columns.tolist()  # np. ['H','E_E1','E_E2','E_E3','E_E4','E_E5']
    metadata = {
        "scaler": scaler,
        "feature_names": feature_names
    }
    joblib.dump(metadata, "scaler.pkl")

    print("Trenowanie zakończone. Zapisano pliki: models.pkl, scaler.pkl (dla Q=0.15, 0.35, 0.5, 0.6)")

if __name__ == "__main__":
    main()
