import json
import pandas as pd
import re
import numpy as np
from io import StringIO

def procesar_reservas_complejo(json_data):
    # Cargar datos en DataFrame
    df = pd.DataFrame(json_data)

    # ==========================================
    # 1. PREPARACIÓN Y FUNCIONES AUXILIARES
    # ==========================================

    # Mapeo de meses en español
    meses = {
        'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
        'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
        'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'
    }

    def extraer_fecha(texto):
        if not isinstance(texto, str):
            return None
        # Regex para: "31 de diciembre del 2013" o "31 diciembre 2013"
        patron = r'(\d{1,2})\s+de\s+([a-zA-Z]+)\s+(?:de|del)\s+(\d{4})'
        match = re.search(patron, texto.lower())
        if match:
            dia, mes_txt, anio = match.groups()
            if mes_txt in meses:
                return f"{int(dia):02d}/{meses[mes_txt]}/{anio}"
        return None

    def normalizar_numero(val):
        if pd.isna(val) or val == '' or str(val).strip() in ['-', '–', '—']:
            return np.nan
        try:
            # Si viene como string, quitar comas de miles si las hubiera y espacios
            if isinstance(val, str):
                val = val.replace(',', '').replace('%', '').strip()
            return float(val)
        except:
            return np.nan

    # Limpieza inicial de filas de unidades (MSTB, MMSCF, etc.)
    # Se asume que si la CATEGORIA es solo una unidad, se descarta.
    unidades_ignorar = ['MSTB', 'MMSCF', 'MBBL', 'BCF']
    df = df[~df['CATEGORIA'].astype(str).isin(unidades_ignorar)]

    # Estructuras para almacenar datos procesados
    # Clave primaria: (fecha, lote, fluido) -> diccionario de valores
    data_reservas = {} 
    # Clave primaria para indicadores: (fecha, lote) -> diccionario de indicadores
    data_indicadores = {}

    # ==========================================
    # 2. ITERACIÓN Y CLASIFICACIÓN
    # ==========================================
    
    # Columnas de fluido a procesar y su mapeo al nombre de salida
    mapa_fluidos = {
        'PETROLEO': 'PETROLEO',
        'GAS': 'GAS',
        'PETROLEO_3': 'LGN' # Asumiendo PETROLEO_3 como LGN/Condensado según estructura común
    }

    for _, row in df.iterrows():
        escenario = str(row.get('ESCENARIO', '')).strip().upper()
        raw_cat = str(row.get('CATEGORIA', '')).strip()
        cat_lower = raw_cat.lower()
        
        # 1. Obtener Fecha y Lote
        fecha_txt = row.get('FECHA_EFECTIVA_TXT', '')
        fecha = extraer_fecha(fecha_txt)
        lote = row.get('LOTE', '')
        if pd.isna(lote) or lote == 'nan': 
            lote = '' # Normalizar lote vacío
        
        if not fecha:
            continue # Si no hay fecha válida, se ignora según regla, o se deja vacío pero agrupado. Asumimos ignorar si es clave.

        key_ind = (fecha, lote)

        # ------------------------------------------
        # LOGICA INDICADORES
        # ------------------------------------------
        if escenario == 'INDICADORES':
            # Normalizar etiqueta
            etiqueta = raw_cat.replace(':', '').upper()
            
            mapa_ind = {
                'IMR': 'IMR', 'IRR': 'IRR', 'ICR': 'ICR', 'IDR': 'IDR',
                'IAR': 'IAR', 'IAR RC': 'IAR_2', 'IAR_RC': 'IAR_2',
                'FR ACTUAL': 'FR actual', 'FR FINAL': 'FR Final'
            }
            
            if etiqueta in mapa_ind:
                col_destino = mapa_ind[etiqueta]
                # Buscar valor: Prioridad derecha (columnas numéricas), luego izquierda (difícil en iterrows, asumimos derecha primero)
                val = np.nan
                # Buscamos en las columnas de valores en orden
                cols_check = ['PETROLEO', 'GAS', 'PETROLEO_3', 'PETROLEO_2']
                found = False
                for col_name in cols_check:
                    v = row.get(col_name)
                    if pd.notna(v) and v != '':
                        val = normalizar_numero(v)
                        found = True
                        break
                
                # Si no encontró a la derecha, lógica "izquierda" (en este JSON plano, suele ser FECHA_EFECTIVA si fuese numérico, pero es texto).
                # Asumimos que si no está en las columnas de fluido, no hay valor.
                
                if found:
                    if key_ind not in data_indicadores:
                        data_indicadores[key_ind] = {}
                    data_indicadores[key_ind][col_destino] = val
            continue

        # ------------------------------------------
        # LOGICA RESERVAS / CONTINGENTES / PROSPECTIVOS
        # ------------------------------------------
        
        # Determinar qué columna de salida corresponde según la categoría y escenario
        target_metric = None
        
        if escenario == 'RESERVAS':
            if 'probadas desarrolladas en producción' in cat_lower: target_metric = 'PDP'
            elif 'probadas desarrolladas en no producción' in cat_lower: target_metric = 'PDNP'
            elif 'probadas desarrolladas' in cat_lower and 'no' not in cat_lower and 'producción' not in cat_lower: 
                pass # Es PD total, lo calcularemos, pero si viene dato lo guardamos como temp
            elif 'probadas no desarrolladas' in cat_lower: target_metric = 'PND'
            elif 'probadas' in cat_lower and ('1p' in cat_lower or cat_lower.endswith('probadas')): target_metric = '1P'
            elif 'probables' in cat_lower: target_metric = 'Probables'
            elif 'posibles' in cat_lower: target_metric = 'Posibles'
            # Nota: 2P y 3P se suelen calcular, pero si vienen explícitos:
            elif '2p' in cat_lower: target_metric = '2P_input' # Guardamos temporal
            elif '3p' in cat_lower: target_metric = '3P_input' # Guardamos temporal

        elif escenario == 'CONTINGENTES':
            if 'estimación baja' in cat_lower or '1c' in cat_lower: target_metric = '1C'
            elif 'mejor estimación' in cat_lower or '2c' in cat_lower: target_metric = '2C'
            elif 'estimación alta' in cat_lower or '3c' in cat_lower: target_metric = '3C'

        elif escenario == 'PROSPECTIVOS':
            if 'estimación baja' in cat_lower or '1u' in cat_lower: target_metric = '1U'
            elif 'mejor estimación' in cat_lower or '2u' in cat_lower: target_metric = '2U'
            elif 'estimación alta' in cat_lower or '3u' in cat_lower: target_metric = '3U'
        
        else:
            continue # Ignorar otros escenarios (PRONOSTICO, etc)

        if target_metric:
            # Iterar sobre los fluidos disponibles en la fila
            for col_json, nombre_fluido in mapa_fluidos.items():
                valor = normalizar_numero(row.get(col_json))
                
                # Solo procesar si hay valor (aunque sea 0, pero no NaN)
                # Ojo: La regla dice "Valores vacíos o guiones -> celda vacía". 
                # normalizar_numero devuelve NaN si es vacío.
                if pd.notna(valor):
                    key_res = (fecha, lote, nombre_fluido)
                    if key_res not in data_reservas:
                        data_reservas[key_res] = {}
                    data_reservas[key_res][target_metric] = valor

    # ==========================================
    # 3. CONSOLIDACIÓN Y CÁLCULOS
    # ==========================================
    
    rows_output = []
    
    # Columnas finales requeridas
    cols_finales = [
        'FECHA.AÑO', 'LOTE', 'FLUIDO', 
        'PDP', 'PDNP', 'PD', 'PND', '1P', 'Probables', '2P', 'Posibles', '3P',
        '1C', '2C', '3C', '1U', '2U', '3U',
        'IMR', 'IRR', 'ICR', 'IDR', 'IAR', 'IAR_2', 'FR actual', 'FR Final'
    ]

    for (fecha, lote, fluido), metricas in data_reservas.items():
        row_out = {col: '' for col in cols_finales}
        row_out['FECHA.AÑO'] = fecha
        row_out['LOTE'] = lote
        row_out['FLUIDO'] = fluido
        
        # Rellenar métricas recolectadas
        for k, v in metricas.items():
            # Si es métrica directa
            if k in cols_finales:
                row_out[k] = v
        
        # --- CÁLCULOS DERIVADOS (Reglas 1.3, 1.7, 1.9 adaptadas) ---
        # Helper para obtener valor o None
        def get_val(key):
            return metricas.get(key)

        # PD = PDP + PDNP (Si ambos existen)
        pdp = get_val('PDP')
        pdnp = get_val('PDNP')
        if pdp is not None and pdnp is not None:
            row_out['PD'] = pdp + pdnp
        
        # 1P (Si viene de input se usa, sino PD + PND)
        p1 = get_val('1P')
        pnd = get_val('PND')
        pd_val = row_out['PD'] if row_out['PD'] != '' else None
        
        # Si no tenemos 1P directo, intentamos calcularlo (aunque la regla dice mapear P1 -> 1P, asumimos prioridad al dato leído)
        if p1 is None and pd_val is not None and pnd is not None:
            # Nota: La regla dice "Probadas (P1) se convertirá en...". 
            # Si no existe, no dice explícitamente calcular, pero en lógica de reservas: 1P = PD + PND
            pass 

        # 2P = 1P + Probables
        probables = get_val('Probables')
        # Usamos el 1P que tengamos (leído o calculado, aqui usamos el leido map)
        if p1 is not None and probables is not None:
            row_out['2P'] = p1 + probables
        elif '2P_input' in metricas: # Si venía del JSON original como "Probadas + Probables"
            row_out['2P'] = metricas['2P_input']

        # 3P = 2P + Posibles
        p2_val = row_out['2P'] if row_out['2P'] != '' else None
        posibles = get_val('Posibles')
        
        if p2_val is not None and posibles is not None:
            row_out['3P'] = p2_val + posibles
        elif '3P_input' in metricas:
            row_out['3P'] = metricas['3P_input']

        # --- INDICADORES ---
        # Buscar indicadores para este (fecha, lote)
        if (fecha, lote) in data_indicadores:
            inds = data_indicadores[(fecha, lote)]
            for k_ind, v_ind in inds.items():
                if k_ind in cols_finales:
                    row_out[k_ind] = v_ind

        rows_output.append(row_out)

    # Crear DataFrame final
    df_final = pd.DataFrame(rows_output, columns=cols_finales)
    
    # Ordenar por fecha y lote
    # Convertir fecha a datetime temporalmente para ordenar
    df_final['temp_date'] = pd.to_datetime(df_final['FECHA.AÑO'], format='%d/%m/%Y', errors='coerce')
    df_final = df_final.sort_values(by=['temp_date', 'LOTE', 'FLUIDO'])
    df_final = df_final.drop(columns=['temp_date'])

    # Generar CSV con punto y coma
    return df_final.to_csv(index=False, sep=',', float_format='%.4f')

# ==========================================
# FUNCIONES DE PROCESAMIENTO MASIVO
# ==========================================

def procesar_archivo(input_file, output_file):
    """Procesa un solo archivo JSON y genera el CSV correspondiente."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            json_input = json.load(f)
        
        csv_result = procesar_reservas_complejo(json_input)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(csv_result)
        
        return True, None
    except FileNotFoundError:
        return False, f"No se encontró el archivo '{input_file}'"
    except json.JSONDecodeError as e:
        return False, f"JSON inválido: {e}"
    except Exception as e:
        return False, f"Error inesperado: {e}"

def procesar_directorio(input_dir, output_dir=None):
    """Procesa todos los archivos JSON en un directorio."""
    import os
    from pathlib import Path
    
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"❌ Error: El directorio '{input_dir}' no existe")
        return
    
    if not input_path.is_dir():
        print(f"❌ Error: '{input_dir}' no es un directorio")
        return
    
    # Si no se especifica directorio de salida, usar el mismo que el de entrada
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Buscar todos los archivos JSON
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"⚠️  No se encontraron archivos JSON en '{input_dir}'")
        return
    
    print(f"📂 Procesando {len(json_files)} archivo(s) JSON...")
    print(f"📁 Directorio de entrada: {input_path}")
    print(f"📁 Directorio de salida: {output_path}")
    print("-" * 60)
    
    exitosos = 0
    fallidos = 0
    
    for json_file in json_files:
        # Generar nombre del archivo CSV (reemplazar .json por .csv)
        csv_filename = json_file.stem + ".csv"
        csv_file = output_path / csv_filename
        
        print(f"⏳ Procesando: {json_file.name}...", end=" ")
        
        success, error = procesar_archivo(json_file, csv_file)
        
        if success:
            print(f"✅ OK → {csv_filename}")
            exitosos += 1
        else:
            print(f"❌ ERROR")
            print(f"   {error}")
            fallidos += 1
    
    print("-" * 60)
    print(f"✅ Completado: {exitosos} exitoso(s), {fallidos} fallido(s)")

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("=" * 60)
        print("TRANSFORMADOR DE RESERVAS - Modo Individual o Masivo")
        print("=" * 60)
        print("\n📌 Uso 1 - Procesar un solo archivo:")
        print("  python transformar_reservas.py <archivo.json> [salida.csv]")
        print("\n📌 Uso 2 - Procesar directorio completo (MASIVO):")
        print("  python transformar_reservas.py --dir <directorio_entrada> [directorio_salida]")
        print("\n📝 Ejemplos:")
        print("  python transformar_reservas.py input.json output.csv")
        print("  python transformar_reservas.py --dir C:\\datos\\json")
        print("  python transformar_reservas.py --dir C:\\datos\\json C:\\datos\\csv")
        sys.exit(1)
    
    # Modo directorio (masivo)
    if sys.argv[1] == "--dir":
        if len(sys.argv) < 3:
            print("❌ Error: Debes especificar el directorio de entrada")
            print("Uso: python transformar_reservas.py --dir <directorio_entrada> [directorio_salida]")
            sys.exit(1)
        
        input_dir = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        
        procesar_directorio(input_dir, output_dir)
    
    # Modo archivo individual
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "output.csv"
        
        print(f"📄 Procesando archivo individual...")
        success, error = procesar_archivo(input_file, output_file)
        
        if success:
            print(f"✅ Transformación completada con éxito!")
            print(f"📁 Archivo CSV generado: {output_file}")
        else:
            print(f"❌ Error: {error}")
            sys.exit(1)