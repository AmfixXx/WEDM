import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import lxml.etree as ET
import re
from io import StringIO
import math

# 1) Ustawiamy layout na 'centered' (strona nie będzie super-szeroka).
st.set_page_config(layout="centered")

# 2) Opcjonalnie możemy wstrzyknąć CSS, aby ograniczyć max-szerokość kontenera:
st.markdown(
    """
    <style>
    /* Ograniczamy główny kontener do 1200px i wycentrowanie */
    .main {
        max-width: 1200px;
        margin: 0 auto;
    }
    /* Możesz też manipulować innymi klasami, np. .css-18e3th9 w zależności od wersji Streamlit */
    </style>
    """,
    unsafe_allow_html=True
)

sns.set(style="whitegrid")

###############################################################################
# 1. Ładowanie modeli i skalera
###############################################################################
@st.cache_resource
def load_model_and_scaler(models_path: str, scaler_path: str):
    models = {}
    scaler = None
    feature_names = []

    try:
        models = joblib.load(models_path)
        # Usuwamy komunikaty st.success(...)
    except FileNotFoundError:
        st.error(f"Plik modeli nie został znaleziony: {models_path}")
    except Exception as e:
        st.error(f"Błąd ładowania modeli: {e}")

    try:
        obj = joblib.load(scaler_path)
        if isinstance(obj, dict) and "scaler" in obj and "feature_names" in obj:
            scaler = obj["scaler"]
            feature_names = obj["feature_names"]
        else:
            scaler = obj
    except FileNotFoundError:
        st.error(f"Plik scaler nie został znaleziony: {scaler_path}")
    except Exception as e:
        st.error(f"Błąd ładowania scaler: {e}")

    return models, scaler, feature_names

###############################################################################
# 2. Funkcja do przewidywania czasu i prędkości
###############################################################################
def predict_time_and_speed(df_input: pd.DataFrame,
                           models: dict,
                           scaler,
                           feature_names: list):
    if scaler is None or not models:
        st.error("Model lub scaler nie został prawidłowo załadowany.")
        return None, None

    if "Dlugosc (mm)" not in df_input.columns:
        st.error("Brakuje kolumny 'Dlugosc (mm)' – nie można policzyć czasu.")
        return None, None

    length_series = df_input["Dlugosc (mm)"].copy()

    rename_map = {
        "Wysokosc": "H",
        "E1": "E_E1",
        "E2": "E_E2",
        "E3": "E_E3",
        "E4": "E_E4",
        "E5": "E_E5",
    }
    df_mapped = df_input.rename(columns=rename_map).copy()

    needed_cols = ["H","E_E1","E_E2","E_E3","E_E4","E_E5"]
    for col in needed_cols:
        if col not in df_mapped.columns:
            df_mapped[col] = 0

    for c in list(df_mapped.columns):
        if c not in needed_cols:
            df_mapped.drop(columns=[c], inplace=True)

    if feature_names:
        final_order = [c for c in feature_names if c in df_mapped.columns]
    else:
        final_order = needed_cols
    df_mapped = df_mapped[final_order]

    try:
        X_scaled = scaler.transform(df_mapped)
    except Exception as e:
        st.error(f"Błąd skalowania cech: {e}")
        return None, None

    rename_map_time = {
        0.15: "Bardzo złe warunki cięcia",
        0.35: "Złe warunki cięcia",
        0.5 : "Standardowe warunki cięcia",
        0.6 : "Idealne warunki cięcia"
    }

    results_df_time = pd.DataFrame(index=df_input.index)
    results_df_speed = pd.DataFrame(index=df_input.index)

    for q_val in sorted(models.keys()):
        model_q = models[q_val]
        try:
            pred_speed = model_q.predict(X_scaled)
        except Exception as e:
            st.error(f"Błąd predict dla kwantyla {q_val}: {e}")
            pred_speed = np.array([0]*len(X_scaled))

        # min = mm / (mm/min)
        times_min = []
        for i, s_val in enumerate(pred_speed):
            if s_val <= 0:
                times_min.append(np.inf)
            else:
                times_min.append(length_series.iloc[i] / s_val)

        def to_h_m(m):
            if np.isinf(m):
                return "∞"
            hh = int(m)//60
            mm = int(m)%60
            return f"{hh}h {mm}m"

        times_h_m = [to_h_m(m) for m in times_min]
        col_time = rename_map_time.get(q_val, f"Kwantyl {q_val}")
        col_speed = f"speed - {col_time}"

        results_df_time[col_time] = times_h_m
        results_df_speed[col_speed] = pred_speed.round(4)

    return results_df_time, results_df_speed

###############################################################################
# 3. Funkcje do parsowania pliku
###############################################################################
def calculate_perimeter(iso_data: str) -> float:
    perimeter = 0.0
    current_pos = (0.0, 0.0)
    start_pos = None

    move_re = re.compile(r'G0?1\s+X([-\d.]+)\s+Y([-\d.]+)', re.IGNORECASE)
    arc_re  = re.compile(r'G0?[23]\s+X([-\d.]+)\s+Y([-\d.]+)\s+I([-\d.]+)\s+J([-\d.]+)', re.IGNORECASE)

    lines = iso_data.split('\n')
    for line in lines:
        l = line.strip()
        if not l or l.startswith(';'):
            continue

        m_move = move_re.match(l)
        if m_move:
            x, y = map(float, m_move.groups())
            dist = math.hypot(x - current_pos[0], y - current_pos[1])
            perimeter += dist
            current_pos = (x, y)
            if start_pos is None:
                start_pos = (x, y)
            continue

        m_arc = arc_re.match(l)
        if m_arc:
            x, y, i, j = map(float, m_arc.groups())
            center = (current_pos[0] + i, current_pos[1] + j)
            r = math.hypot(current_pos[0] - center[0], current_pos[1] - center[1])
            start_ang = math.atan2(current_pos[1] - center[1], current_pos[0] - center[0])
            end_pos = (x, y)
            end_ang = math.atan2(end_pos[1], end_pos[0])
            diff = end_ang - start_ang
            if 'G02' in l.upper():
                if diff > 0:
                    diff -= 2*math.pi
            else:
                if diff < 0:
                    diff += 2*math.pi

            arc_len = abs(r * diff)
            perimeter += arc_len
            current_pos = end_pos
            if start_pos is None:
                start_pos = end_pos

    if start_pos and (current_pos != start_pos):
        perimeter += math.hypot(start_pos[0] - current_pos[0], start_pos[1] - current_pos[1])

    return perimeter

def process_uploaded_file(file_content: bytes):
    try:
        try:
            decoded = file_content.decode("utf-8")
        except UnicodeDecodeError:
            decoded = file_content.decode("windows-1252")

        parser = ET.XMLParser(recover=True)
        root = ET.fromstring(decoded.encode("utf-8"), parser=parser)

        shapes = {}
        shapes_node = root.find('Shapes')
        if shapes_node is not None:
            for shape_el in shapes_node.findall('Shape'):
                shape_name = shape_el.findtext('ShapeName','').strip()
                geoms_el = shape_el.find('Geometries')
                if not geoms_el:
                    continue
                total_len = 0.0
                for geom in geoms_el.findall('Geometry'):
                    iso_el = geom.find('ISOData')
                    if iso_el is not None and iso_el.text:
                        perimeter_val = calculate_perimeter(iso_el.text)
                        total_len += perimeter_val
                if total_len > 0:
                    shapes[shape_name] = shapes.get(shape_name, 0.0) + total_len

        cut_ops = []
        programs_node = root.find('Programs')
        if programs_node is not None:
            for prog_el in programs_node.findall('Program'):
                ops_el = prog_el.find('Operations')
                if ops_el is None or not ops_el.text:
                    continue
                for line in ops_el.text.split('\n'):
                    l = line.strip()
                    if l.startswith('CUT,'):
                        parts = l.split(',')
                        if len(parts) < 5:
                            continue
                        _, fig_raw, seq_raw, pass_raw, height_raw = parts[:5]
                        fig_name = fig_raw.strip()
                        pass_type = pass_raw.strip()
                        h_str = height_raw.replace('H','').strip()
                        try:
                            h_val = float(h_str)
                        except:
                            h_val = 0.0
                        length_val = shapes.get(fig_name, 0.0)
                        cut_ops.append({
                            'FigureName': fig_name,
                            'Sequence': seq_raw.strip(),
                            'PassType': pass_type,
                            'Height_mm': h_val,
                            'Perimeter_mm': length_val
                        })

        if not cut_ops:
            return None

        input_rows = []
        for op in cut_ops:
            row = {}
            row['Wysokosc'] = op['Height_mm']
            d_rounded = round(op['Perimeter_mm'], 1)
            row['Dlugosc (mm)'] = d_rounded
            row['ilosc_nastaw'] = 1

            e_dict = {f"E{i}": 0 for i in range(1,11)}
            pass_up = op['PassType'].upper()
            pass_pref = pass_up.split('_')[0]
            if pass_pref.startswith('E'):
                if pass_pref in e_dict:
                    e_dict[pass_pref] = 1
            row.update(e_dict)
            input_rows.append(row)

        if not input_rows:
            return None

        input_df = pd.DataFrame(input_rows)

        # Usuwamy E6..E10
        for i in range(6, 11):
            col = f"E{i}"
            if col in input_df.columns:
                input_df.drop(columns=[col], inplace=True)

        st.subheader("Dane wejściowe")

        # Dodajemy kolumnę "Obróbka" i przenosimy ją na przód
        input_df.insert(0, "Obróbka", range(1, len(input_df)+1))

        display_cols = ["Obróbka","Wysokosc", "Dlugosc (mm)", "ilosc_nastaw", "E1", "E2", "E3", "E4", "E5"]
        existing_cols = [c for c in display_cols if c in input_df.columns]
        input_df = input_df[existing_cols]

        # Wyświetlamy tabelę na np. 1000px szerokości (i domyślną wysokość)
        st.dataframe(input_df, width=1000)

        return input_df

    except Exception as e:
        return None

###############################################################################
# 4. Główna funkcja
###############################################################################
def main():
    st.title("Szacowanie czasu obróbki WEDM")

    models_path = "models.pkl"
    scaler_path = "scaler.pkl"
    models, scaler, feature_names = load_model_and_scaler(models_path, scaler_path)

    st.header("Wybierz metodę wprowadzania danych")
    method = st.radio("", ["Wczytaj plik MJB/XML/EJB", "Wprowadź dane ręcznie"])

    if method == "Wczytaj plik MJB/XML/EJB":
        st.subheader("Wczytaj plik MJB/XML/EJB")
        up_file = st.file_uploader(
            "Przeciągnij i upuść plik (max ~200MB)",
            type=["mjb","xml","ejb"]
        )
        if up_file is not None:
            st.write(f"Plik: {up_file.name}, rozmiar: {up_file.size/1024:.2f} kB")
            input_df = process_uploaded_file(up_file.read())
            if input_df is not None and not input_df.empty:
                if st.button("Szacuj czas obróbki"):
                    with st.spinner("Przetwarzanie..."):
                        results_time, results_speed = predict_time_and_speed(
                            input_df, models, scaler, feature_names
                        )
                    if results_time is not None:
                        # Kolumna "Obróbka" w wynikach
                        results_time.insert(0, "Obróbka", range(1, len(results_time)+1))
                        results_speed.insert(0, "Obróbka", range(1, len(results_speed)+1))

                        st.subheader("Wyniki - Czas [h m]")
                        st.dataframe(results_time, width=1000)

                        st.subheader("Wyniki - Prędkość [mm/min]")
                        st.dataframe(results_speed, width=1000)

                        sums = []
                        for col in results_time.columns:
                            if col == "Obróbka":
                                continue
                            total_minutes = 0.0
                            for val in results_time[col]:
                                if val == "∞":
                                    total_minutes = np.inf
                                    break
                                parts = val.split()
                                hh = int(parts[0].replace("h",""))
                                mm = int(parts[1].replace("m",""))
                                total_minutes += hh*60 + mm
                            sums.append((col, total_minutes))

                        st.subheader("Szacowane czasy:")
                        for (col_name, total_m) in sums:
                            if np.isinf(total_m):
                                st.write(f"- {col_name}: ∞")
                            else:
                                hh = int(total_m)//60
                                mm = int(total_m)%60
                                st.write(f"- {col_name}: {hh}h {mm}m")

    else:
        st.subheader("Wpisz dane ręcznie")

        col1, col2 = st.columns(2)
        with col1:
            wysokosc_val = st.number_input("Wysokość (mm):", min_value=0, value=30)
        with col2:
            dlugosc_val = st.number_input("Długość ścieżki (mm):", min_value=0, value=200)

        st.write("Zaznacz nastawy (E1..E5)")
        ccols = st.columns(5)
        e_bool = []
        for i in range(5):
            lab = f"E{i+1}"
            val = ccols[i].checkbox(lab, value=False)
            e_bool.append(val)

        ilosc_nastaw = sum(e_bool)

        row_dict = {
            "Wysokosc": wysokosc_val,
            "Dlugosc (mm)": round(dlugosc_val, 1),
            "ilosc_nastaw": ilosc_nastaw,
            "E1": 1 if e_bool[0] else 0,
            "E2": 1 if e_bool[1] else 0,
            "E3": 1 if e_bool[2] else 0,
            "E4": 1 if e_bool[3] else 0,
            "E5": 1 if e_bool[4] else 0
        }

        df_manual = pd.DataFrame([row_dict])
        st.write("Dane wejściowe")

        df_manual.insert(0, "Obróbka", [1])

        disp_cols = ["Obróbka","Wysokosc","Dlugosc (mm)","ilosc_nastaw","E1","E2","E3","E4","E5"]
        exist_cols = [c for c in disp_cols if c in df_manual.columns]
        df_manual = df_manual[exist_cols]

        st.dataframe(df_manual, width=1000)

        if st.button("Szacuj czas obróbki"):
            with st.spinner("Przetwarzanie..."):
                results_time, results_speed = predict_time_and_speed(
                    df_manual, models, scaler, feature_names
                )
            if results_time is not None:
                results_time.insert(0, "Obróbka", range(1, len(results_time)+1))
                results_speed.insert(0, "Obróbka", range(1, len(results_speed)+1))

                st.subheader("Wyniki - Czas [h m]")
                st.dataframe(results_time, width=1000)

                st.subheader("Wyniki - Prędkość [mm/min]")
                st.dataframe(results_speed, width=1000)

                sums = []
                for col_name in results_time.columns:
                    if col_name == "Obróbka":
                        continue
                    total_m = 0.0
                    for val in results_time[col_name]:
                        if val == "∞":
                            total_m = np.inf
                            break
                        parts = val.split()
                        hh = int(parts[0].replace("h",""))
                        mm = int(parts[1].replace("m",""))
                        total_m += hh*60 + mm
                    sums.append((col_name, total_m))

                st.subheader("Szacowane czasy:")
                for (col_name, total_m) in sums:
                    if np.isinf(total_m):
                        st.write(f"- {col_name}: ∞")
                    else:
                        hh = int(total_m)//60
                        mm = int(total_m)%60
                        st.write(f"- {col_name}: {hh}h {mm}m")

    st.markdown("---")


if __name__ == "__main__":
    main()
