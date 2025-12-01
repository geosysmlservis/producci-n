import json
import pandas as pd
import re
import os
import sys
from pathlib import Path
from io import StringIO

def procesar_reservas(json_data):
    # Cargar datos
    df = pd.DataFrame(json_data)

    # 1. Limpieza inicial
    # Eliminar filas donde YACIMIENTO es nulo, "nan", "NOTA:" o textos informativos
    df = df[df['YACIMIENTO'].notna()]
    df = df[~df['YACIMIENTO'].astype(str).isin(['nan', 'NOTA:', 'Se consideran dentro el pronóstico...'])]
    
    # 2. Convertir columnas de años (2014, 2015...) de columnas a filas (Unpivot/Melt)
    # Identificamos las columnas que son años (numéricas o strings numéricos)
    year_cols = [c for c in df.columns if c.isdigit()]
    id_vars = ['HOJA', 'TIPO', 'FECHA_EFECTIVA_TXT', 'CATEGORIA', 'YACIMIENTO']
    
    df_melted = df.melt(id_vars=id_vars, value_vars=year_cols, var_name='ANIO_PRON', value_name='VALOR')
    
    # Convertir valores a numérico, forzando 0 si es nulo
    df_melted['VALOR'] = pd.to_numeric(df_melted['VALOR'], errors='coerce').fillna(0.0)

    # 3. Pivotar para tener las CATEGORIAS como columnas
    # Indice único: Fecha Efectiva, Yacimiento, Año Pronóstico
    df_pivot = df_melted.pivot_table(
        index=['FECHA_EFECTIVA_TXT', 'YACIMIENTO', 'ANIO_PRON'], 
        columns='CATEGORIA', 
        values='VALOR', 
        aggfunc='sum'
    ).reset_index()

    # Rellenar NaN con 0.0 tras el pivot
    df_pivot = df_pivot.fillna(0.0)

    # 4. Mapeo de columnas según reglas del usuario
    # Definir nombres internos estandarizados para facilitar cálculos
    col_mapping = {
        'Probadas Desarrolladas en Producción': 'PDP(BOPD)',
        'Probadas Desarrolladas en No Producción': 'PDNP(BOPD)', # Si existiera
        'Probadas No Desarrolladas': 'PND(BOPD)',
        'Probables': 'probables',
        'Posibles': 'posibles',
        'Recurso Contingente': 'RC(BOPD)',
        'Recurso Prospectivo': 'RP(BOPD)'
        # Nota: 'Total Probadas' y 'Total Probadas Desarrolladas' se ignoran para recalcular
    }

    # Renombrar las columnas existentes
    df_pivot = df_pivot.rename(columns=col_mapping)

    # Asegurar que existan todas las columnas base (incluso si no venían en el JSON)
    required_cols = ['PDP(BOPD)', 'PDNP(BOPD)', 'PND(BOPD)', 'probables', 'posibles', 'RC(BOPD)', 'RP(BOPD)']
    for col in required_cols:
        if col not in df_pivot.columns:
            df_pivot[col] = 0.0

    # 5. Cálculos Matemáticos (Reglas 1.3, 1.5, 1.7, 1.9)
    
    # 1.3 PD = PDP + PDNP
    df_pivot['PD'] = df_pivot['PDP(BOPD)'] + df_pivot['PDNP(BOPD)']
    
    # 1.5 1P = PD + PND
    df_pivot['1P'] = df_pivot['PD'] + df_pivot['PND(BOPD)']
    
    # 1.7 2P = 1P + probables
    df_pivot['2P'] = df_pivot['1P'] + df_pivot['probables']
    
    # 1.9 3P = 2P + posibles
    df_pivot['3P'] = df_pivot['2P'] + df_pivot['posibles']

    # 6. Manejo de Fechas (Reglas 2.1 y 2.2)

    # Función para parsear fecha efectiva en español
    def parse_spanish_date(date_str):
        if not isinstance(date_str, str): return ""
        # Diccionario de meses
        meses = {
            'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
            'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
            'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'
        }
        try:
            # Buscar patrón: "mes dia, año" o similar
            # Ejemplo entrada: "efectivo: diciembre 31, 2013"
            match = re.search(r'([a-z]+)\s+(\d{1,2})[,\s]+(\d{4})', date_str.lower())
            if match:
                mes_txt, dia, anio = match.groups()
                if mes_txt in meses:
                    return f"{int(dia):02d}/{meses[mes_txt]}/{anio}"
            return date_str # Retornar original si falla
        except:
            return date_str

    df_pivot['FECHA.AÑO'] = df_pivot['FECHA_EFECTIVA_TXT'].apply(parse_spanish_date)

    # Crear FECHA.AÑO.PRON (Regla 2.1)
    # Se asume fin de año para las columnas de año (ej: 2014 -> 31/12/2014)
    df_pivot['FECHA.AÑO.PRON'] = df_pivot['ANIO_PRON'].apply(lambda x: f"31/12/{x}")

    # 7. Selección y Ordenamiento final
    final_cols = [
        'FECHA.AÑO',
        'FECHA.AÑO.PRON',
        'YACIMIENTO',
        'PDP(BOPD)',
        'PDNP(BOPD)',
        'PD',
        'PND(BOPD)',
        '1P',
        'probables',
        '2P',
        'posibles',
        '3P',
        'RC(BOPD)',
        'RP(BOPD)'
    ]
    
    df_final = df_pivot[final_cols].copy()

    # 8. Formato numérico (Regla 3.2: double, truncado a 4 decimales)
    numeric_cols = final_cols[3:] # Todas desde PDP(BOPD)
    
    for col in numeric_cols:
        # Asegurar float y formatear
        df_final[col] = df_final[col].apply(lambda x: f"{float(x):.4f}")

    # Generar CSV string
    csv_output = df_final.to_csv(index=False, sep=',')
    return csv_output

def procesar_archivo(input_file, output_file):
    """Procesa un solo archivo JSON y genera el CSV correspondiente."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            json_input = json.load(f)
        
        csv_result = procesar_reservas(json_input)
        
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

# --- Bloque de ejecución principal ---
if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("=" * 60)
        print("TRANSFORMADOR DE PRONÓSTICOS - Modo Individual o Masivo")
        print("=" * 60)
        print("\n📌 Uso 1 - Procesar un solo archivo:")
        print("  python transformar_pronostico.py <archivo.json> [salida.csv]")
        print("\n📌 Uso 2 - Procesar directorio completo (MASIVO):")
        print("  python transformar_pronostico.py --dir <directorio_entrada> [directorio_salida]")
        print("\n📝 Ejemplos:")
        print("  python transformar_pronostico.py input.json output.csv")
        print("  python transformar_pronostico.py --dir C:\\datos\\json")
        print("  python transformar_pronostico.py --dir C:\\datos\\json C:\\datos\\csv")
        sys.exit(1)
    
    # Modo directorio (masivo)
    if sys.argv[1] == "--dir":
        if len(sys.argv) < 3:
            print("❌ Error: Debes especificar el directorio de entrada")
            print("Uso: python transformar_pronostico.py --dir <directorio_entrada> [directorio_salida]")
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