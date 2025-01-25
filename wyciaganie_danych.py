import os
import re
import math
import uuid  # <-- Do generowania unikalnych ID
import xml.etree.ElementTree as ET

import csv
import openpyxl  # do obsługi zapisów XLSX


def parse_iso_line(line):
    """
    Prosty parser linii G-kodu typu G01, G02, G03:
      - Wyciąga X, Y, I, J (jeśli występują).
      - Rozpoznaje czy to ruch liniowy (G1) czy łuk (G2/G3).
    Zwraca dict: {
       'type': 'G1'/'G2'/'G3',  # ewentualnie 'G' gdy G0
       'x': float lub None,
       'y': float lub None,
       'i': float lub None,
       'j': float lub None
    }
    """
    import re

    g_match = re.search(r'(G0|G1|G2|G3|G01|G02|G03)\s?', line, re.IGNORECASE)
    if not g_match:
        return None

    # Normalizujemy: G01 -> G1, G02->G2, G03->G3, G00->G
    g_code = g_match.group(1).upper()
    # Zamiana np. G01 -> G1, G02 -> G2, G03->G3, G00->G
    g_code = g_code.replace('0', '')

    x_match = re.search(r'X([\-\d\.]+)', line, re.IGNORECASE)
    y_match = re.search(r'Y([\-\d\.]+)', line, re.IGNORECASE)
    i_match = re.search(r'I([\-\d\.]+)', line, re.IGNORECASE)
    j_match = re.search(r'J([\-\d\.]+)', line, re.IGNORECASE)

    x_val = float(x_match.group(1)) if x_match else None
    y_val = float(y_match.group(1)) if y_match else None
    i_val = float(i_match.group(1)) if i_match else None
    j_val = float(j_match.group(1)) if j_match else None

    return {
        'type': g_code,
        'x': x_val,
        'y': y_val,
        'i': i_val,
        'j': j_val,
    }


def compute_contour_length(iso_data):
    """
    Oblicza nominalny obwód figury na podstawie G-kodu w iso_data (typowo w <ISOData>).
    - Rozpoznaje G1 (ruch liniowy), G2/G3 (ruch łukowy).
    - W uproszczeniu (zakładamy G2/G3 z parametrami I,J - środek łuku).
    Zwraca float - całkowita długość (w jednostkach G-kodu, zwykle mm).
    """
    lines = iso_data.strip().split('\n')
    current_x, current_y = 0.0, 0.0
    length_sum = 0.0

    for line in lines:
        parsed = parse_iso_line(line)
        if not parsed:
            continue

        g_type = parsed['type']  # 'G1','G2','G3' lub 'G'
        x_new = parsed['x']
        y_new = parsed['y']
        i_val = parsed['i']
        j_val = parsed['j']

        # Ruch prostoliniowy (G1 lub G)
        if g_type in ['G1', 'G'] and (x_new is not None) and (y_new is not None):
            dist = math.hypot(x_new - current_x, y_new - current_y)
            length_sum += dist
            current_x, current_y = x_new, y_new

        # Ruch łukowy (G2/G3)
        elif g_type in ['G2', 'G3'] and (x_new is not None) and (y_new is not None) \
                and (i_val is not None) and (j_val is not None):

            cx = current_x + i_val
            cy = current_y + j_val
            r = math.hypot(i_val, j_val)

            start_angle = math.atan2(current_y - cy, current_x - cx)
            end_angle   = math.atan2(y_new - cy, x_new - cx)

            dtheta = end_angle - start_angle
            if g_type == 'G2':
                # G2 -> ruch w kierunku CW
                while dtheta > 0:
                    dtheta -= 2*math.pi
            else:
                # G3 -> ruch w kierunku CCW
                while dtheta < 0:
                    dtheta += 2*math.pi

            arc_length = abs(r * dtheta)
            length_sum += arc_length
            current_x, current_y = x_new, y_new

    return length_sum


def parse_shapes(root):
    """
    Parsuje sekcję <Shapes> i liczy sumaryczny obwód figur:
      - { shapeName: obwod_float }
    """
    shape_dict = {}
    shapes_node = root.find('Shapes')
    if shapes_node is not None:
        for shape_el in shapes_node.findall('Shape'):
            name_el = shape_el.find('ShapeName')
            if name_el is None:
                continue
            shape_name = name_el.text.strip()

            geo_nodes = shape_el.find('Geometries')
            if not geo_nodes:
                shape_dict[shape_name] = 0.0
                continue

            total_length = 0.0
            for geo_el in geo_nodes.findall('Geometry'):
                iso_el = geo_el.find('ISOData')
                if iso_el is not None and iso_el.text:
                    length_val = compute_contour_length(iso_el.text)
                    total_length += length_val

            shape_dict[shape_name] = total_length

    return shape_dict


def parse_operations_and_durations(root):
    """
    Z sekcji <TraceDataCtx> -> <histo> -> <operation block="CUT, ...">
      wyciąga:
        - figure_name (np. "Fig_XXXX")
        - E (np. E1/E2/E3)
        - H (wysokość)
        - machiningDuration (float, w sekundach!)
    Zwraca listę słowników:
      [
        {
          'figure': ...,
          'E': ...,
          'H': ...,
          'machiningDuration': ...
        },
        ...
      ]
    """
    results = []
    trace_node = root.find('TraceDataCtx')
    if trace_node is None:
        return results

    all_main_tags = trace_node.findall('.//MainTag')
    for main_tag in all_main_tags:
        histo = main_tag.find('histo')
        if histo is None:
            continue
        operations = histo.findall('.//operation')
        for op in operations:
            block_attr = op.get('block', '')
            if block_attr.startswith('CUT,'):
                # np. "CUT,Fig_007_10,Sequence 5,E2,H120.0"
                mch_dur_str = op.get('machiningDuration', '0')
                try:
                    mch_dur = float(mch_dur_str)  # W SEKUNDACH
                except:
                    mch_dur = 0.0

                cut_parts = block_attr.split(',')
                figure_name = ""
                E_val = None
                H_val = None

                if len(cut_parts) >= 2:
                    figure_name = cut_parts[1].strip()

                for cpart in cut_parts[2:]:
                    cpart = cpart.strip()
                    # np. "E2" albo "H35.0"
                    e_match = re.match(r'(E\d+)', cpart, re.IGNORECASE)
                    if e_match:
                        E_val = e_match.group(1)

                    h_match = re.match(r'H([\-\d\.]+)', cpart, re.IGNORECASE)
                    if h_match:
                        try:
                            H_val = float(h_match.group(1))
                        except:
                            H_val = 0.0

                results.append({
                    'figure': figure_name,
                    'E': E_val,
                    'H': H_val,
                    'machiningDuration': mch_dur
                })

    return results


def process_mjb_file(filepath):
    """
    Wczytuje i parsuje plik XML (mjb/xml/ejb).
    Zwraca:
      shape_dict = { figura: obwód },
      op_list = [ { figure, E, H, machiningDuration }, ... ]
    """
    tree = ET.parse(filepath)  # może rzucić ParseError
    root = tree.getroot()

    shape_dict = parse_shapes(root)
    op_list = parse_operations_and_durations(root)

    return shape_dict, op_list


def main_parser(output_csv='wynik.csv', output_xlsx='wynik.xlsx'):
    """
    Główny skrypt:
      - rekurencyjnie przechodzi folder base_dir
      - dopuszcza pliki .mjb .xml .ejb
      - parsuje i zbiera dane do CSV + XLSX
      - do każdego pliku przypisuje unikalny ID
      - oblicza prędkość (mm/min) przy założeniu, że machiningDuration jest w sekundach
    """
    base_dir = "C:/Users/Szymon/Desktop/projekt/Czasy"

    # Przygotowujemy strukturę wierszy + nazwy kolumn (bez filename)
    rows = []
    columns = [
        'uniqueID',            # Unikalny ID na plik
        'figure',
        'E',
        'H',
        'perimeter',           # mm
        'machiningDuration_s', # sekundy
        'speed_mm_min'
    ]

    for dirpath, dirnames, filenames in os.walk(base_dir):
        for fname in filenames:
            # Filtr plików
            if not (fname.lower().endswith('.mjb') or
                    fname.lower().endswith('.xml') or
                    fname.lower().endswith('.ejb')):
                continue

            fullpath = os.path.join(dirpath, fname)
            print(f"--> Analyzing file: {fname}")

            # Generujemy unikalny ID RAZ na plik
            file_id = str(uuid.uuid4())

            # Próbujemy sparsować plik
            try:
                shape_dict, op_list = process_mjb_file(fullpath)
            except ET.ParseError as e:
                print(f"   !! ParseError w pliku: {fname}\n      {e}")
                # pomijamy plik
                continue

            # Dla każdej operacji przypisujemy ten sam file_id
            for op in op_list:
                figure = op['figure']
                e_val  = op['E']
                h_val  = op['H']
                duration_s = op['machiningDuration']  # w sekundach
                perimeter_mm = shape_dict.get(figure, 0.0)

                # Obliczamy prędkość w mm/min:
                #   speed = perimeter / (duration_s / 60) = perimeter * 60 / duration_s
                if duration_s > 0:
                    speed_mm_min = perimeter_mm * 60.0 / duration_s
                else:
                    speed_mm_min = 0.0

                row = [
                    file_id,
                    figure,
                    e_val,
                    h_val,
                    perimeter_mm,
                    duration_s,
                    speed_mm_min
                ]
                rows.append(row)

    # Zapis do CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(columns)
        for r in rows:
            writer.writerow(r)

    # Zapis do XLSX
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Data"
    ws.append(columns)
    for r in rows:
        ws.append(r)

    wb.save(output_xlsx)
    print("Zakończono przetwarzanie. Pliki wynikowe:", output_csv, "oraz", output_xlsx)


if __name__ == "__main__":
    main_parser("wynik.csv", "wynik.xlsx")
