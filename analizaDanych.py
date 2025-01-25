import pandas as pd
import numpy as np

def remove_outliers(df, column, lower_quantile=0.3, upper_quantile=0.6):
    """
    Usuwa outliers z kolumny 'column' w DataFrame 'df',
    biorąc pod uwagę percentyle lower_quantile i upper_quantile.
    Zwraca df bez wartości poza tym zakresem.
    """
    q_low = df[column].quantile(lower_quantile)
    q_high = df[column].quantile(upper_quantile)
    mask = (df[column] >= q_low) & (df[column] <= q_high)
    return df[mask].copy()

def main():
    # 1. Wczytanie pliku z danymi, np. "wynik.csv"
    input_csv = "wynik.csv"
    df = pd.read_csv(input_csv, delimiter=';')  # dostosuj delimiter

    # 2. Usunięcie outliers w kolumnie speed_mm_min (zakres 0.3..0.6)
    df_clean = remove_outliers(df, 'speed_mm_min', 0.3, 0.6)

    # 3. Obliczanie wybranych kwantyli: [0.15, 0.35, 0.5, 0.6]
    quantiles_to_compute = [0.15, 0.35, 0.5, 0.6]

    # Grupujemy (H, E) i liczymy kwantyle w speed_mm_min
    grouped = df_clean.groupby(['H', 'E'])['speed_mm_min'].quantile(quantiles_to_compute)

    # 4. Zamieniamy w Series -> DataFrame z kolumnami [H, E, Q, speed_mm_min]
    result_df = grouped.reset_index()
    # Pandas nada kolumnie z kwantylami nazwę "level_2", zamienimy ją na "Q"
    result_df.rename(columns={'level_2': 'Q', 'speed_mm_min': 'speed_mm_min'}, inplace=True)

    # 5. Zapis do plików CSV i XLSX z nazwą "kwantyle3.*"
    output_csv = "kwantyle3.csv"
    result_df.to_csv(output_csv, sep=';', index=False)

    output_xlsx = "kwantyle3.xlsx"
    result_df.to_excel(output_xlsx, sheet_name="Kwantyle3", index=False)

    print("Zapisano wynik w:", output_csv, "oraz", output_xlsx)

if __name__ == "__main__":
    main()
