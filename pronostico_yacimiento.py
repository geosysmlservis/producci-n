# -*- coding: utf-8 -*-
import os
import re
import math
import argparse
from typing import Any, Dict, List, Tuple, Optional
import json

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ---------------------------
# Utilidades básicas
# ---------------------------
REG_COLS = ["archivo", "procesado", "fecha_procesamiento", "intentos", "estado"]
REG_NAME = "registro_procesamiento.csv"

CATEGORIAS_RESERVAS = [
    "Probadas Desarrolladas en Producción",
    "Desarrolladas en Producción",
    "Total Probadas Desarrolladas",
    "Probadas No Desarrolladas",
    "Total Probadas",
    "Probables",
    "Posibles",
    "TOTAL"
]

def _yymmdd_today() -> str:
    return datetime.now().isoformat()

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _scan_input_files(root: Path, exts=(".xlsx", ".csv")) -> list[Path]:
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(exts):
                files.append(Path(dp) / fn)
    return sorted(files)

def _load_or_init_registry(out_dir: Path) -> pd.DataFrame:
    _ensure_dir(out_dir)
    reg_path = out_dir / REG_NAME
    if reg_path.exists():
        df = pd.read_csv(reg_path)
        for c in REG_COLS:
            if c not in df.columns:
                if c == "fecha_procesamiento": df[c] = ""
                elif c == "procesado": df[c] = False
                elif c == "estado": df[c] = "pendiente"
                elif c == "intentos": df[c] = 0
                elif c == "archivo": df[c] = ""
        return df[REG_COLS]
    return pd.DataFrame(columns=REG_COLS)

def _save_registry(df: pd.DataFrame, out_dir: Path) -> Path:
    reg_path = out_dir / REG_NAME
    df.to_csv(reg_path, index=False)
    return reg_path

def _upsert_pending_entry(reg: pd.DataFrame, key: str) -> pd.DataFrame:
    if reg.empty or not (reg["archivo"] == key).any():
        reg = pd.concat([reg, pd.DataFrame([{
            "archivo": key, "procesado": False,
            "fecha_procesamiento": "", "intentos": 0, "estado": "pendiente"
        }])], ignore_index=True)
    return reg

def _increment_attempt(reg: pd.DataFrame, key: str) -> pd.DataFrame:
    m = reg["archivo"] == key
    s = reg.loc[m, "intentos"].fillna(0).infer_objects(copy=False)
    reg.loc[m, "intentos"] = s.astype(int) + 1
    return reg

def _mark_success(reg: pd.DataFrame, key: str) -> pd.DataFrame:
    m = reg["archivo"] == key
    reg.loc[m, "procesado"] = True
    reg.loc[m, "estado"] = "pendiente"
    reg.loc[m, "fecha_procesamiento"] = _yymmdd_today()
    return reg

def _mark_error(reg: pd.DataFrame, key: str, set_date: bool = False) -> pd.DataFrame:
    m = reg["archivo"] == key
    reg.loc[m, "procesado"] = False
    reg.loc[m, "estado"] = "error"
    if set_date:
        reg.loc[m, "fecha_procesamiento"] = _yymmdd_today()
    return reg

# nombre de salida TXT consolidado por documento
def _json_output_path(out_dir: Path, rel_path: str) -> Path:
    rel = Path(rel_path)
    dest_dir = out_dir / rel.parent
    dest_name = f"{rel.stem}_pronostico.json"
    dest = dest_dir / dest_name
    _ensure_dir(dest.parent)
    return dest


# utilidades de datos

def _es_num(v: object) -> bool:
    if isinstance(v, (int, float, np.integer)) and pd.notna(v):
        return True
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    s = str(v).strip().replace("\xa0", " ")
    s = s.replace(",", "")           # quitar miles
    s = re.sub(r"\s+", "", s)
    if s in {"", "-", "–", "—"}:
        return False
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", s))

def _norm_dato(x: object) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = str(x).lower().strip()
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def _parsear_anio(v):
    # acepta numérico entero o string año 19xx/20xx
    if isinstance(v, (int, np.integer, float)) and pd.notna(v) and float(v).is_integer():
        yi = int(v)
        if 1900 <= yi <= 2100:
            return yi
    s = _norm_dato(v)
    m = re.fullmatch(r"(19|20)\d{2}", s)
    return int(s) if m else None

def _numify(s: pd.Series) -> pd.Series:
    s = (s.astype(str)
           .str.replace("\xa0", " ", regex=False)
           .str.replace(",", "", regex=False)   # miles
           .str.strip()
           .replace({"": np.nan, "-": np.nan, "–": np.nan, "—": np.nan}))
    return pd.to_numeric(s, errors="coerce")

def _abs_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols):
        df.loc[:, num_cols] = df.loc[:, num_cols].abs()
    return df

def _extraer_categorias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica filas que son categorías (headers/totales) en la columna YACIMIENTO,
    mueve ese valor a una nueva columna CATEGORIA y propaga el valor a las filas hijas.
    Limpia el valor de YACIMIENTO para las filas que son categorías.
    """
    if "YACIMIENTO" not in df.columns:
        return df

    # Normalizar para búsqueda (opcional, pero seguro)
    # Aquí asumimos coincidencia exacta o muy cercana.
    # Usaremos una lista de valores lower para comparar.
    cats_lower = {c.lower(): c for c in CATEGORIAS_RESERVAS}
    
    # Nueva columna
    df["CATEGORIA"] = None
    
    current_cat = None
    
    # Iterar por índices para poder modificar
    for idx in df.index:
        val = df.at[idx, "YACIMIENTO"]
        if not val or pd.isna(val):
            # Si está vacío, solo asignamos la categoría actual si existe
            if current_cat:
                df.at[idx, "CATEGORIA"] = current_cat
            continue
            
        val_str = str(val).strip()
        val_lower = val_str.lower()
        
        # Chequear si es una categoría
        match_cat = None
        if val_lower in cats_lower:
            match_cat = cats_lower[val_lower]
        
        if match_cat:
            # Es una línea de categoría (header o total)
            current_cat = match_cat
            df.at[idx, "CATEGORIA"] = match_cat
            df.at[idx, "YACIMIENTO"] = None # Limpiar YACIMIENTO
        else:
            # Es un dato (hijo)
            if current_cat:
                df.at[idx, "CATEGORIA"] = current_cat
                
    return df

# ---------------------------
# Detectores específicos de "Cuadro N° ... por Yacimiento"
# ---------------------------

CUADRO_RX = re.compile(
    r"\bcuadro\s*(n[°o]|no\.?)\s*\d+\s*:\s*pron(ó|o)stico\s+de\s+producci(ó|o)n\s+de\s+"
    r"(condensados|gas\s+natural|glp)\s+por\s+yacimiento\b",
    re.I
)

TIME_HEADER_RX = re.compile(r"(a(ñ|n)o|anio|mes|fecha|periodo|período|^q[1-4]$|^\d{4}(-\d{2})?$)", re.I)


def _tipo_from_commodity(commodity: str) -> str:
    c = (commodity or "").strip().lower()
    # Mapear a los tres tipos pedidos: HIDROCARBUROS, GAS, GLP
    if "glp" in c:
        return "PRONOSTICO_GLP"
    if "gas natural" in c or re.search(r"\bgas\b", c):
        return "PRONOSTICO_GAS"
    if any(k in c for k in ["hidrocarburo", "condensado", "crudo", "liquido", "líquido", "petroleo", "petróleo"]):
        return "PRONOSTICO_HIDROCARBUROS"
    return "PRONOSTICO"

def _infer_tipo_from_headers(headers: List[str]) -> str:
    text = " ".join([str(h).lower() for h in headers])
    if re.search(r"\bglp\b", text):
        return "PRONOSTICO_GLP"
    if re.search(r"\bgas(\s+natural)?\b", text):
        return "PRONOSTICO_GAS"
    if re.search(r"hidrocarburo|condensado|crudo|l[ií]quido|petr[oó]leo", text):
        return "PRONOSTICO_HIDROCARBUROS"
    return "PRONOSTICO"



def detectar_cuadros_pronostico(df: pd.DataFrame, scan_rows: int = 80, scan_cols: int = 20):
    R = min(scan_rows, df.shape[0])
    C = min(scan_cols, df.shape[1])
    hallados = []
    for r in range(R):
        for c in range(C):
            s = str(df.iat[r, c] if c < df.shape[1] else "").replace("\xa0"," ").strip()
            if not s:
                continue
            m = CUADRO_RX.search(s.lower())
            if m:
                hallados.append({
                    "fila": r,
                    "col": c,
                    "commodity": m.group(3).lower().replace("  ", " ")
                })
    return hallados


def _buscar_fila_header_tiempo(df: pd.DataFrame, r_ini: int, max_seek: int = 6):
    for r in range(r_ini+1, min(df.shape[0], r_ini+1+max_seek)):
        row = df.iloc[r].astype(str).fillna("").str.strip()
        for c, val in enumerate(row):
            if TIME_HEADER_RX.search(val.lower()):
                headers = row.tolist()
                headers = [h if h else f"col_{i}" for i, h in enumerate(headers)]
                return r, headers
    return None, None


def extraer_tabla_desde_cuadro(df: pd.DataFrame, hoja: str) -> pd.DataFrame:
    cuadros = detectar_cuadros_pronostico(df)
    tablas = []
    for cu in cuadros:
        r_hdr, headers = _buscar_fila_header_tiempo(df, cu["fila"])
        if r_hdr is None:
            continue
        if r_hdr is None or headers is None:
            # Si no encontramos fila header o no hay headers válidos, saltar este cuadro
            continue
        body = df.iloc[r_hdr+1:].copy()
        body.columns = [h if h else f"col_{i}" for i, h in enumerate(headers)]

        # eje de tiempo
        time_col = None
        for h in body.columns[:5]:
            if TIME_HEADER_RX.search(str(h)):
                time_col = h; break
        if time_col is None:
            time_col = body.columns[0]

        body = body[body[time_col].astype(str).str.strip() != ""]
        if body.empty:
            continue

        # numéricos
        for i, c in enumerate(body.columns):
            if c != time_col:
                # Si es la primera columna y no es la de tiempo, asumimos que es etiqueta (YACIMIENTO, etc.)
                # y no la forzamos a numérico.
                if i == 0 and c != time_col:
                    continue
                body[c] = pd.to_numeric(body[c], errors="coerce")

        commodity = cu["commodity"]
        tipo = _tipo_from_commodity(commodity)
        _tipo_cols = _infer_tipo_from_headers(list(body.columns))
        if tipo == "PRONOSTICO" and _tipo_cols != "PRONOSTICO":
            tipo = _tipo_cols

        body.insert(0, "TIPO", tipo)
        body.insert(0, "COMMODITY", commodity)
        body.insert(0, "HOJA", hoja)

        tablas.append(body)

    if not tablas:
        return pd.DataFrame()
    return pd.concat(tablas, ignore_index=True, sort=False)

# ---------------------------
# Núcleo del algoritmo (genérico)
# ---------------------------

def detectar_fila_anios(df: pd.DataFrame, secuencia: int = 5, max_rows: int = 120
                        ) -> Tuple[int, int, List[int], List[int]]:
    candidato = (None, None, [], [])
    ROWS = min(max_rows, df.shape[0])

    for row in range(ROWS):
        sec_init, col_init, pos_init = [], None, []
        for col, v in enumerate(df.iloc[row, :]):
            anio = _parsear_anio(v)
            if anio is not None:
                if not sec_init:
                    sec_init, col_init, pos_init = [anio], col, [col]
                else:
                    if anio == sec_init[-1] + 1:
                        sec_init.append(anio)
                        pos_init.append(col)
                    else:
                        if len(sec_init) > len(candidato[2]):
                            candidato = (row, col_init, sec_init.copy(), pos_init.copy())
                        sec_init, col_init, pos_init = [anio], col, [col]
            else:
                if len(sec_init) > len(candidato[2]):
                    candidato = (row, col_init, sec_init.copy(), pos_init.copy())
                sec_init, col_init, pos_init = [], None, []
        if len(sec_init) > len(candidato[2]):
            candidato = (row, col_init, sec_init.copy(), pos_init.copy())

    if not candidato[2] or len(candidato[2]) < secuencia:
        raise ValueError("No se encontró una secuencia de años válida.")
    if candidato[0] is None or candidato[1] is None:
        raise ValueError("No se encontró una secuencia de años válida.")
    print(f"Secuencia de años encontrada en fila {candidato[0]}: {candidato[2]}")
    return int(candidato[0]), int(candidato[1]), candidato[2], candidato[3]


def detectar_col_texto(df: pd.DataFrame, pos_anios: List[int], fila_anios: int,
                       profundidad: int = 60, umbral_texto: float = 0.6
                       ) -> Tuple[int, pd.Series, List[str], List[int]]:
    col_init = pos_anios[0] - 1
    fila_init = fila_anios + 1
    fila_end = min(df.shape[0], fila_init + profundidad)

    # Fecha Efectiva
    fecha_efectiva = []
    for r in range(fila_init, -1, -1):
        for c in range(col_init + 1):
            texto = _norm_dato(df.iat[r, c])
            if "efectiv" in texto:
                fecha_efectiva.append(texto)
                break
        if fecha_efectiva:
            break

    # Columna de rótulos
    for c in range(col_init, -1, -1):
        col_vals = df.iloc[fila_init:fila_end, c]
        non_empty = [v for v in col_vals if _norm_dato(v) != ""]
        if not non_empty:
            continue
        text_like = [v for v in non_empty if not _es_num(v)]
        frac_text = len(text_like) / len(non_empty)
        if frac_text >= umbral_texto:
            serie = df.iloc[fila_init:, c].astype(str).map(lambda s: re.sub(r"\s+", " ", s.strip()))
            return c, serie, fecha_efectiva, list(serie.index)

    # Fallback
    for c in range(col_init, -1, -1):
        col_vals = df.iloc[fila_init:fila_end, c]
        if any(_norm_dato(v) != "" for v in col_vals):
            serie = df.iloc[fila_init:, c].astype(str).map(lambda s: re.sub(r"\s+", " ", s.strip()))
            return c, serie, fecha_efectiva, list(serie.index)

    raise ValueError("No se encontró una columna de rótulo a la izquierda del primer año.")


def detectar_fila_final(df: pd.DataFrame, fila_anios: int, c_izq: int,
                        cols_anio: List[int], vacias_tol: int = 3) -> int:
    c2 = max(cols_anio)
    vacias = 0
    last = fila_anios
    for r in range(fila_anios + 1, df.shape[0]):
        fila = df.iloc[r, c_izq:c2+1]
        hay_algo = any(str(v).strip() not in {"", "-", "–", "—"} for v in fila)
        if hay_algo:
            last = r; vacias = 0
        else:
            vacias += 1
            if vacias >= vacias_tol:
                break
    return last


def submatriz_por_posiciones(df: pd.DataFrame,
                             fila_anios: int,
                             fila_fin: int,
                             c_izq: int,
                             c_der: int,
                             cols_anio: List[int],
                             nombres_rotulos: Optional[List[str]] = None,
                             fecha_txt: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    idx_cols = list(range(c_izq, c_der + 1)) + list(cols_anio)
    bloque = df.iloc[fila_anios:fila_fin + 1, idx_cols].copy()

    k = c_der - c_izq + 1
    if not nombres_rotulos or len(nombres_rotulos) != k:
        nombres_rotulos = [f"label_{i+1}" for i in range(k)]

    years = []
    for c in cols_anio:
        raw = str(df.iat[fila_anios, c]).strip()
        m = re.search(r"(19|20)\d{2}", raw)
        years.append(m.group(0) if m else raw)

    headers = nombres_rotulos + [str(y) for y in years]
    minw = min(len(headers), bloque.shape[1])
    bloque = bloque.iloc[:, :minw]
    bloque.columns = headers[:minw]

    bloque = bloque.iloc[1:, :].reset_index(drop=True)
    for name in nombres_rotulos:
        if name in bloque.columns:
            bloque[name] = bloque[name].astype(str).str.strip().replace({"": np.nan})
    for y in years:
        col = str(y)
        if col in bloque.columns:
            bloque[col] = _numify(bloque[col])

    if fecha_txt is not None:
        bloque.insert(0, "FECHA_EFECTIVA_TXT", str(fecha_txt))

    meta = {
        "row_years": fila_anios,
        "row_end": fila_fin,
        "label_cols": list(range(c_izq, c_der + 1)),
        "year_cols": cols_anio,
        "years": years,
        "fecha_efectiva_txt": fecha_txt
    }
    return bloque, meta

# ---------------------------
# Orquestación por hoja
# ---------------------------

def procesa_hoja(df: pd.DataFrame, sheet_name: str,
                 keyword: str = "pronósti") -> Optional[pd.DataFrame]:
    # 0) Prioriza cuadros explícitos (Condensados / Gas Natural / GLP por Yacimiento)
    t_cuadro = extraer_tabla_desde_cuadro(df, sheet_name)
    if t_cuadro is not None and not t_cuadro.empty:
        # Aplicar extracción de categorías también aquí
        return _extraer_categorias(t_cuadro)

    # 1) ¿contiene keyword?
    plano = _norm_dato(" ".join(df.astype(str).fillna("").values.ravel().tolist()))
    has_kw = (not keyword) or (keyword in plano)


    # 2) detectar fila de años
    fila_anio, col0, anios, pos_anios = detectar_fila_anios(df, secuencia=5, max_rows=150)

    # 3) columna de rótulos + fecha efectiva
    col_yac, serie_yac, fechas_txt, _ = detectar_col_texto(df, pos_anios, fila_anio)

    # 4) detectar fin de bloque y construir submatriz
    c_izq = c_der = col_yac
    fila_fin = detectar_fila_final(df, fila_anio, c_izq, pos_anios)

    fecha_txt = next((t for t in fechas_txt if str(t).strip()), "")
    sub, _meta = submatriz_por_posiciones(
        df=df,
        fila_anios=fila_anio,
        fila_fin=fila_fin,
        c_izq=c_izq,
        c_der=c_der,
        cols_anio=pos_anios,
        nombres_rotulos=["YACIMIENTO"],
        fecha_txt=fecha_txt
    )
    # Añade contexto y TIPO genérico si no vino por cuadro
    sub.insert(0, "HOJA", sheet_name)
    # Si no vino tipo (no entró por 'Cuadro …'), inferir por headers
    if "TIPO" not in sub.columns:
        _inferred = _infer_tipo_from_headers(list(sub.columns))
        sub.insert(1, "TIPO", _inferred)

    # 5) Extraer categorías de YACIMIENTO
    sub = _extraer_categorias(sub)

    return sub

# ---------------------------
# Runner principal
# ---------------------------

def extraer_pronosticos(
    input_xlsx: str,
    out_dir: str,
    keyword: str = "pronósti",
    per_doc_txt: Optional[str] = None   # consolidado por documento
) -> List[str]:
    if not os.path.exists(input_xlsx):
        raise FileNotFoundError(f"No existe: {input_xlsx}")
    os.makedirs(out_dir, exist_ok=True)

    xls = pd.ExcelFile(input_xlsx, engine="openpyxl")
    generados = []
    colecta = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(input_xlsx, sheet_name=sheet, header=None, dtype=object, engine="openpyxl")
        try:
            sub = procesa_hoja(df, str(sheet), keyword=keyword)
            if sub is None or sub.empty:
                continue
            colecta.append(sub)
        except Exception as e:
            print(f"[WARN] Hoja '{sheet}': {e}")
            continue

    if colecta:
        doc = pd.concat(colecta, ignore_index=True, sort=False)
        # columnas de contexto al frente
        preferidas = [c for c in ["HOJA","TIPO","COMMODITY","FECHA_EFECTIVA_TXT","CATEGORIA","YACIMIENTO"] if c in doc.columns]
        otras = [c for c in doc.columns if c not in preferidas]
        doc = doc[preferidas + otras]
        doc = _abs_numeric(doc)  # fuerza positivos

        per_doc_json = per_doc_txt or os.path.join(out_dir, f"{Path(input_xlsx).stem}.json")
        doc.to_json(per_doc_json, orient="records", force_ascii=False, indent=None) #opción 1 un json compacto
        #doc.to_json(per_doc_json, orient='records', force_ascii=False, indent=2) opción 2
        generados.append(per_doc_json)

    return generados

# ---------------------------
# Indexado y procesamiento en lote
# ---------------------------

def indexar_entradas_generar_registro(input_dir: str, out_dir: str) -> Path:
    in_root = Path(input_dir).resolve()
    out_root = Path(out_dir).resolve()
    files = _scan_input_files(in_root, exts=(".xlsx", ".csv"))
    reg = _load_or_init_registry(out_root)

    for src in files:
        rel = os.path.relpath(src, start=in_root)
        rel_norm = rel.replace("\\", "/")
        json_path = _json_output_path(out_root, rel_norm)
        key = json_path.name
        reg = _upsert_pending_entry(reg, key)
        if not json_path.exists():
            json_path.write_text(
                f"ORIGEN: {src}\nGENERADO: {datetime.now().isoformat(timespec='seconds')}\nESTADO_INICIAL: pendiente\n",
                encoding="utf-8"
            )
    return _save_registry(reg, out_root)


def procesar_lote(input_dir: str, out_dir: str,
                  keyword: str = "pronósti") -> Path:
    in_root = Path(input_dir).resolve()
    out_root = Path(out_dir).resolve()
    files = _scan_input_files(in_root, exts=(".xlsx", ".csv"))
    reg = _load_or_init_registry(out_root)

    for src in files:
        rel = os.path.relpath(src, start=in_root)
        rel_norm = rel.replace("\\", "/")
        json_path = _json_output_path(out_root, rel_norm)
        key = json_path.name
        reg = _upsert_pending_entry(reg, key)

        reg = _increment_attempt(reg, key)
        try:
            salidas = extraer_pronosticos(
                input_xlsx=str(src),
                out_dir=str(out_root),
                keyword=keyword,
                per_doc_txt=str(json_path)
            )
            ok = bool(salidas)
            reg = _mark_success(reg, key) if ok else _mark_error(reg, key)
        except Exception:
            reg = _mark_error(reg, key)

    return _save_registry(reg, out_root)


# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pronóstico de yacimientos (modo masivo con registro).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    one = sub.add_parser("one", help="Procesa un único Excel/CSV")
    one.add_argument("--input_xlsx", required=True)
    one.add_argument("-o","--out_dir", default="salida_pronosticos")
    one.add_argument("--keyword", default="pronósti")

    idx = sub.add_parser("index", help="Indexa recursivamente y crea registro + TXT stub")
    idx.add_argument("--input_dir", required=True)
    idx.add_argument("--out_dir", required=True)

    proc = sub.add_parser("process", help="Procesa recursivamente y actualiza el registro")
    proc.add_argument("--input_dir", required=True)
    proc.add_argument("--out_dir", required=True)
    proc.add_argument("--keyword", default="pronósti")

    args = ap.parse_args()

    if args.cmd == "one":
        txt_path = Path(args.out_dir) / f"{Path(args.input_xlsx).stem}.json"
        outs = extraer_pronosticos(args.input_xlsx, args.out_dir, keyword=args.keyword, per_doc_txt=str(txt_path))
        print("\n".join(outs) if outs else "No se detectaron tablas de pronóstico.")
    elif args.cmd == "index":
        reg = indexar_entradas_generar_registro(args.input_dir, args.out_dir)
        print(f"[OK] Registro en: {reg}")
    elif args.cmd == "process":
        reg = procesar_lote(args.input_dir, args.out_dir, keyword=args.keyword)
        print(f"[OK] Registro actualizado en: {reg}")
