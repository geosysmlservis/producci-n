#
"""
Reservorios, Contingentes, Prospectivos e Indicadores — versión optimizada + INDEXADOR RECURSIVO
----------------------------------------------------------------------------------------------
Novedades:
- Escaneo recursivo de carpetas (--input_dir) para .xlsx y .csv
- Generación de 1 TXT por **documento** encontrado (manteniendo estructura de subcarpetas en out_dir)
- Registro global único CSV: archivo, procesado, fecha_procesamiento (YYYY/MM/DD), intentos, estado
- Modo "index" (solo indexa y crea TXT por documento) y modo "process" (intenta procesar y marcar ok/error)
- Preserva el flujo previo por archivo único (--input) y sus salidas por hoja / unificado

Nota: Este archivo integra todo el código previo y añade utilidades de indexación/registro.
"""
from __future__ import annotations
import os
import re
import math
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional, Callable
import json

import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# Configuración
# =========================
@dataclass
class CFG:
    HEAD_ROWS: int = 60         # filas para plano/clasificación
    HEAD_COLS: int = 30         # columnas para plano/clasificación
    HEADER_DEPTH: int = 10       # profundidad de encabezados para detectar fluidos/unidades
    MAX_DOWN: int = 1056        # recorrido hacia abajo para fin de bloque
    EMPTY_TOL: int = 10          # filas vacías consecutivas que determinan fin
    RIGHT_PAD: int = 40         # margen derecho por si no hay ancla
    REG_NAME: str = "registro_procesamiento.csv"

cfg = CFG()

# =========================
# Patrones
# =========================
FLUID_RX = {
    "PETROLEO": re.compile(r"\bpetr[oó]leo(\s+crudo)?\b", re.I),
    "GAS":      re.compile(r"\bgas(\s+natural)?\b", re.I),
    "LGN":      re.compile(r"\blgn\b|\b(l[ií])quidos?\s+de(l)?\s+gas\b", re.I),
}
PAT = {
    "categoria": re.compile(r"\bcategor(í|i)a\b", re.I),
    # antes: r"l(í|i)mite\s+econ(ó|o)mico"
    "limite": re.compile(
        r"(l(í|i)mite\s+econ(ó|o)mico|hasta\s+el\s+fin\s+de\s+la\s+vida\s+(ú|u)til|vida\s+(ú|u)til)",
        re.I
    ),
    "contrato": re.compile(r"t(é|e)rmino\s+del\s+contrato", re.I),
}
UNIT_RX = re.compile(r"\b(mstb?|mbbl|mmbbl|mmscf|mmcf|bn|bbl)\b", re.I)
LOTE_RX = re.compile(r"\bLOTE\s+([0-9IVXLCDM]+)\b", re.I)

# Frases clave (en minúsculas)
KEY_RESERVAS = "resumen de las reservas y recursos"
KEY_CONTING  = "resumen de recursos contingentes"
KEY_PROSPECT = "resumen de recursos prospectivos"
KEY_INDIC    = "indicadores"

# Indicadores gestion
INDICADORES = [
    "IMR", "IRR", "IBR", "ICR", "IDR", "IAR", "FR ACTUAL", "FR FINAL"
]
DECIMAL_DOT_THOUSANDS_COMMA  = re.compile(r'^\d{1,3}(,\d{3})+(\.\d+)?$')      # 123,456.78
DECIMAL_COMMA_THOUSANDS_DOT  = re.compile(r'^\d{1,3}(\.\d{3})+(,\d+)?$')      # 123.456,78

RES_VAR_RX = re.compile(
    r"(petrol|petr[oó]leo|crudo|l[ií]quido|condensad|gas(?!\s*lift)|glp|lgn|"
    r"\bimr\b|\birr\b|\bibr\b|\bicr\b|\bidr\b|\biar\b|fr\s*(actual|final))",
    re.I
)
FECHA_RX = re.compile(r"(al\\s+)?(\\d{1,2}[./-]\\d{1,2}[./-](\\d{2,4}))", re.I)

# =========================
# Utilidades comunes
# =========================
def _norm_dato(x: object) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = str(x).replace("\xa0", " ").lower().strip()
    s = s.replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", s)


def _non_empty_sem(x) -> bool:
    s = str(x).strip().lower()
    return s not in {"", "-", "–", "—", "nan", "none"}


def _plane(df: pd.DataFrame, max_rows: int = cfg.HEAD_ROWS, max_cols: int = cfg.HEAD_COLS) -> str:
    r = min(max_rows, df.shape[0]); c = min(max_cols, df.shape[1])
    vals = df.iloc[:r, :c].astype(str).fillna("").values.ravel(order="C")
    s = " ".join(vals).lower()
    return re.sub(r"\s+", " ", s)


def _flatten_and_make_unique_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["__".join(map(str, t)).strip() for t in df.columns.values]
    else:
        df.columns = [str(c) for c in df.columns]
    seen: Dict[str, int] = {}
    new_cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols
    return df


def _make_unique(cols: List[str]) -> List[str]:
    from collections import defaultdict
    seen = defaultdict(int)
    out: List[str] = []
    for c in cols:
        seen[c] += 1
        out.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    return out

# Numerificación
def _parse_number(x):
    s = str(x).strip().replace('\xa0',' ').replace(' ', '')
    if s == '' or s in {'-', '–', '—', 'nan', 'None', 'NaN'}:
        return np.nan
    # 123.456,78  -> 123456.78
    if DECIMAL_COMMA_THOUSANDS_DOT.match(s):
        s = s.replace('.', '').replace(',', '.')
    # 123,456.78  -> 123456.78
    elif DECIMAL_DOT_THOUSANDS_COMMA.match(s):
        s = s.replace(',', '')
    # 123,45  -> 123.45
    elif (',' in s) and ('.' not in s):
        s = s.replace(',', '.')
    # else: ya está en 123.45
    try:
        return float(s)
    except:
        return np.nan
    
def _numify_series(s: pd.Series) -> pd.Series:
    s = (s.astype(str)
           .str.replace("\xa0", " ", regex=False)
           .str.replace("%", "", regex=False)
           .str.replace(".", "", regex=False)
           .str.replace(",", ".", regex=False)
           .str.strip()
           .replace({"": np.nan, "-": np.nan, "–": np.nan, "—": np.nan}))
    return pd.to_numeric(s, errors="coerce")

def _numify_df(df: pd.DataFrame) -> pd.DataFrame:
    vfunc = np.vectorize(_parse_number)
    return df.apply(vfunc)

def _format_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].apply(
                lambda v: "" if pd.isna(v) else np.format_float_positional(float(v), trim='-')
            )
    return df

def _extraer_fecha_plano(df: pd.DataFrame, head_rows: int = 60, head_cols: int = 25) -> str:
    R, C = min(head_rows, df.shape[0]), min(head_cols, df.shape[1])
    for r in range(R):
        for c in range(C):
            s = str(df.iat[r, c]).replace("\\xa0"," ").strip()
            m = FECHA_RX.search(s)
            if m:
                return m.group(2)
    return ""
# =========================
# Detecciones auxiliares
# =========================
def extraer_lote(df: pd.DataFrame, top_rows: int = 40, left_cols: int = 20) -> str:
    R = min(top_rows, df.shape[0]); C = min(left_cols, df.shape[1])
    for r in range(R):
        for c in range(C):
            m = LOTE_RX.search(str(df.iat[r, c]))
            if m:
                return f"LOTE {m.group(1).upper()}"
    return ""

def extraer_reservas_sin_anios(df: pd.DataFrame, hoja: str) -> Optional[pd.DataFrame]:
    # 1) localizar fila de cabecera con 'CATEGORIA' / 'CATEGORÍA'
    row_hdr, col_cat = None, None
    for r in range(min(80, df.shape[0])):
        vals = df.iloc[r].astype(str).str.strip().str.lower()
        for c, v in enumerate(vals):
            if "categor" in v:  # cubre categoria/categoría
                row_hdr, col_cat = r, c
                break
        if row_hdr is not None: break
    if row_hdr is None: return None

    # 2) columnas de valor (variables) a la derecha del header
    headers = df.iloc[row_hdr].astype(str).fillna("").str.strip().tolist()
    value_idx = [i for i,h in enumerate(headers) if i != col_cat and RES_VAR_RX.search(h)]
    if not value_idx: return None

    # 3) cuerpo de la tabla
    body = df.iloc[row_hdr+1:].copy()
    body.columns = [h if h else f"col_{i}" for i,h in enumerate(headers)]
    sub = body.iloc[:, [col_cat] + value_idx].copy()
    sub = sub[(sub.iloc[:,0].astype(str).str.strip() != "") | (sub.iloc[:,1:].notna().any(axis=1))]
    if sub.empty: return None

    # 4) numerificar variables y extraer fecha
    for j in range(1, sub.shape[1]):
        sub.iloc[:, j] = pd.to_numeric(
            sub.iloc[:, j].astype(str)
              .str.replace("\\xa0"," ", regex=False)
              .str.replace(".", "", regex=False)   # miles estilo 1.234,56
              .str.replace(",", ".", regex=False)  # decimal
              .str.strip(),
            errors="coerce")
        
    fecha_txt = _extraer_fecha_plano(df)

    # 5) normalizar columnas y fill-down de CATEGORIA
    sub.rename(columns={sub.columns[0]: "CATEGORIA"}, inplace=True)
    sub["CATEGORIA"] = sub["CATEGORIA"].replace({"": np.nan}).ffill()
    sub.insert(0, "HOJA", hoja)
    if "FECHA_EFECTIVA_TXT" not in sub.columns:
        sub.insert(1, "FECHA_EFECTIVA_TXT", fecha_txt)
    return sub

def extraer_fecha(df: pd.DataFrame, ventana: Optional[Tuple[int, int, int, int]] = None) -> str:
    if ventana:
        r0, rn, c0, cn = ventana
        for r in range(max(0, r0), min(df.shape[0], rn)):
            fila = " ".join(str(df.iat[r, c]) for c in range(max(0, c0), min(df.shape[1], cn)) if str(df.iat[r, c]).strip())
            if "cuadro" in _norm_dato(fila):
                return _norm_dato(fila)
    # Fallback: primeras 10 filas
    for r in range(min(10, df.shape[0])):
        fila = " ".join(str(df.iat[r, c]) for c in range(df.shape[1]) if _non_empty_sem(df.iat[r, c]))
        if "cuadro" in _norm_dato(fila):
            return _norm_dato(fila)
    return ""


def detectar_fin_bloque(df: pd.DataFrame, r_start: int, c_label: int, val_cols: List[int],
                        empty_tol: int = cfg.EMPTY_TOL, max_down: int = cfg.MAX_DOWN) -> int:
    r_end, vac = r_start, 0
    lim_inf = min(df.shape[0] - 1, r_start + max_down)
    while r_end + 1 <= lim_inf:
        r_end += 1
        left = _norm_dato(df.iat[r_end, c_label])
        if left.startswith("nota"):
            r_end -= 1
            break
        has_cat = _non_empty_sem(df.iat[r_end, c_label])
        has_val = any(_non_empty_sem(df.iat[r_end, j]) for j in val_cols)
        if has_cat or has_val:
            vac = 0
        else:
            vac += 1
            if vac >= empty_tol:
                r_end -= vac
                break
    return r_end


def detectar_columnas_fluido(df: pd.DataFrame, r_top: int, c_left: int, c_right: int,
                             header_depth: int = cfg.HEADER_DEPTH) -> List[Dict[str, Any]]:
    r0 = max(0, r_top); r1 = min(df.shape[0], r0 + 1 + header_depth)
    c0 = max(0, c_left); c1 = min(df.shape[1], c_right + 1)

    win = df.iloc[r0:r1, c0:c1].astype(str).fillna("")
    win = win.map(lambda x: re.sub(r"\s+", " ", x.replace("\xa0", " ").lower().strip()))

    cand: List[Dict[str, Any]] = []
    for rr in range(win.shape[0]):
        for cc in range(win.shape[1]):
            s = win.iat[rr, cc]
            if not s:
                continue
            blobs = [s]
            if rr + 1 < win.shape[0]:
                blobs.append(str(s) + " " + str(win.iat[rr + 1, cc]))
            if cc + 1 < win.shape[1]:
                blobs.append(str(s) + " " + str(win.iat[rr, cc + 1]))
            for b in blobs:
                for name, rx in FLUID_RX.items():
                    if rx.search(str(b)): # Se asegura que 'b' sea un string para el método search
                        unit = ""
                        m = UNIT_RX.search(str(b)) # Se asegura que 'b' sea un string para el método search
                        if not m and rr + 1 < win.shape[0]:
                            m = UNIT_RX.search(str(win.iat[rr + 1, cc]))
                        unit = (m.group(0).upper() if m else "")
                        cand.append({'col': c0 + cc, 'fluido': name, 'unidad': unit})
                        break
    out: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for d in sorted(cand, key=lambda x: x['col']):
        if d['col'] not in seen:
            seen.add(d['col']); out.append(d)
    return out

# =========================
# Clasificación de hoja
# =========================
def clasificar_hoja_por_frase(df: pd.DataFrame) -> Optional[str]:
    plano = _plane(df)
    if KEY_RESERVAS in plano:  return "RESERVAS"
    # Nuevos títulos esperados en tus hojas
    if "estimación de reservas al límite económico" in plano: return "RESERVAS"
    if "estimación de reservas al término del contrato" in plano: return "RESERVAS"
    if "resumen de reservas" in plano: return "RESERVAS"
    if KEY_CONTING  in plano:  return "CONTINGENTES"
    if KEY_PROSPECT in plano:  return "PROSPECTIVOS"
    if KEY_INDIC    in plano:  return "INDICADORES"
    
    return None

# =========================
# Extractor: RESERVAS
# =========================
def detectar_anclas_reservas(df: pd.DataFrame, max_cols: int = 20, max_rows: int = 25) -> Dict[str, Tuple[int,int,str]]:
    # usar el PAT global ya declarado arriba (con 'vida útil')
    patrones = PAT
    posiciones = {"categoria": [], "limite": [], "contrato": []}
    ROWS, COLS = min(max_rows, df.shape[0]), min(max_cols, df.shape[1])
    for r in range(ROWS):
        for c in range(COLS):
            txt = _norm_dato(df.iat[r, c])
            if not txt:
                continue
            for k, rx in patrones.items():
                if rx.search(txt):
                    posiciones[k].append((r, c, txt))
    return {k: min(lst, key=lambda t: (t[0], t[1])) for k, lst in posiciones.items() if lst}


def completar_tabla_reservas(df: pd.DataFrame, pos: Dict[str, Tuple[int, int, str]],
                             filas_arriba: int = 12, filas_abajo: int = 1,
                             ancho_der_extra: int = 20, ancho_izq: int = 0) -> Dict[str, Any]:
    if "categoria" not in pos or "limite" not in pos:
        return {"sub": pd.DataFrame(), "meta": {"msg": "Faltan anclas 'categoria' o 'limite'."}}

    r_cat, c_cat = pos["categoria"][0], pos["categoria"][1]
    r_lim, c_lim = pos["limite"][0], pos["limite"][1]

    # Ventana para fecha
    r0 = max(0, r_cat - filas_arriba)
    rn = min(df.shape[0], r_lim + filas_abajo)
    c0 = max(0, c_cat - ancho_izq)
    cn = min(df.shape[1], pos.get("contrato", (None, c_lim + cfg.RIGHT_PAD))[1] if pos.get("contrato") else c_lim + cfg.RIGHT_PAD)

    lote = extraer_lote(df)
    fecha_txt = extraer_fecha(df, ventana=(r0, rn, c0, cn))

    # Detectar columnas de fluido
    cols_info = detectar_columnas_fluido(
    df,
    r_top=r_lim - 2,     # antes -1; baja una fila más para capturar unidades
    c_left=c_lim,
    c_right=cn,
    header_depth=max(cfg.HEADER_DEPTH, 8)  # asegura 8 niveles
    )

    if not cols_info:
        return {"sub": pd.DataFrame(), "meta": {"msg": "No se detectaron columnas de fluido."}}

    # Coherencia: excluir columna de CATEGORIA si se coló
    cols_info = [d for d in cols_info if d['col'] != c_cat]
    val_cols = [d['col'] for d in cols_info]

    # Inicio de datos
    r_start = r_lim + 1
    if r_start < df.shape[0]:
        vals = pd.Series([_norm_dato(df.iat[r_start, j]) for j in val_cols])
        hits = sum(bool(UNIT_RX.search(v)) for v in vals)
        if hits >= max(1, len(val_cols)//2):
            r_start += 1
    lim_sup = min(df.shape[0] - 1, r_start + 30)
    while r_start <= lim_sup:
        hay_cat = _non_empty_sem(df.iat[r_start, c_cat])
        hay_val = any(_non_empty_sem(df.iat[r_start, j]) for j in val_cols)
        if hay_cat or hay_val:
            break
        r_start += 1

    r_end = detectar_fin_bloque(df, r_start, c_cat, val_cols, empty_tol=cfg.EMPTY_TOL, max_down=cfg.MAX_DOWN)
    if r_end < r_start:
        return {"sub": pd.DataFrame(), "meta": {"msg": "Rango vacío."}}

    sub = df.iloc[r_start:r_end + 1, [c_cat] + val_cols].copy()

    # Encabezados y contexto
    headers = [
        "LOTE",
        "FECHA_EFECTIVA_TXT",
        "CATEGORIA",
    ] + [(f"{d['fluido']}_{d['unidad']}" if d.get('unidad') else d['fluido']) for d in cols_info]
    headers = _make_unique(headers)

    sub.columns = ["CATEGORIA"] + headers[3:]

    # Inserta contexto
    sub.insert(0, "FECHA_EFECTIVA_TXT", fecha_txt)
    sub.insert(0, "LOTE", lote)

    # Numerificar valores
    if sub.shape[1] > 3:
        num_cols = sub.columns[3:]
        sub.loc[:, num_cols] = _numify_df(sub.loc[:, num_cols])

    meta = {
        "rango_filas": (r_start, r_end),
        "fluids": cols_info,
        "ventana": (r0, rn, c0, cn)
    }
    return {"sub": sub, "meta": meta}

# =========================
# Extractor: RECURSOS (Contingentes / Prospectivos)
# =========================
def detectar_anclas_recursos(df: pd.DataFrame, max_rows: int = 12, max_cols: int = 30) -> Dict[str, Any]:
    categoria_pos: Optional[Tuple[int, int]] = None
    ROWS, COLS = min(max_rows, df.shape[0]), min(max_cols, df.shape[1])
    for r in range(ROWS):
        for c in range(COLS):
            if re.search(r"\bcategor(í|i)a\b", _norm_dato(df.iat[r, c])):
                categoria_pos = (r, c)
                break
        if categoria_pos:
            break
    if not categoria_pos:
        return {"error": "No se encontró 'Categoría'."}

    r_cat, c_cat = categoria_pos
    c_right = min(df.shape[1] - 1, c_cat + cfg.RIGHT_PAD)
    cols_info = detectar_columnas_fluido(df, r_top=r_cat, c_left=c_cat + 1, c_right=c_right, header_depth=cfg.HEADER_DEPTH)
    if not cols_info:
        return {"error": "No se detectaron columnas de fluido."}

    return {"categoria_pos": categoria_pos, "cols_info": cols_info}


def extraer_tabla_recursos(df: pd.DataFrame) -> Dict[str, Any]:
    info = detectar_anclas_recursos(df)
    if "error" in info:
        return {"data": pd.DataFrame(), "meta": info}

    (r_cat, c_cat), cols_info = info["categoria_pos"], info["cols_info"]
    val_cols = [d['col'] for d in cols_info]

    r_start = r_cat + 1
    if r_start < df.shape[0]:
        vals = pd.Series([_norm_dato(df.iat[r_start, j]) for j in val_cols])
        hits = sum(bool(UNIT_RX.search(v)) for v in vals)
        if hits >= max(1, len(val_cols)//2):
            r_start += 1
    while r_start < df.shape[0] and not (_non_empty_sem(df.iat[r_start, c_cat]) or any(_non_empty_sem(df.iat[r_start, j]) for j in val_cols)):
        r_start += 1

    r_end = detectar_fin_bloque(df, r_start, c_cat, val_cols, empty_tol=cfg.EMPTY_TOL, max_down=cfg.MAX_DOWN)
    if r_end < r_start:
        return {"data": pd.DataFrame(), "meta": {"msg": "Rango vacío."}}

    sub = df.iloc[r_start:r_end + 1, [c_cat] + val_cols].copy()

    headers = ["CATEGORIA"] + [ (f"{d['fluido']}_{d['unidad']}" if d.get('unidad') else d['fluido']) for d in cols_info ]
    headers = _make_unique(headers)
    sub.columns = headers

    # Contexto
    lote = extraer_lote(df)
    fecha_txt = extraer_fecha(df)
    sub.insert(0, "FECHA_EFECTIVA_TXT", fecha_txt)
    sub.insert(0, "LOTE", lote)

    if sub.shape[1] > 3:
        num_cols = sub.columns[3:]
        sub.loc[:, num_cols] = _numify_df(sub.loc[:, num_cols])

    meta = {"rango_filas": (r_start, r_end), "fluids": cols_info}
    return {"data": sub, "meta": meta}

# =========================
# Extractor: INDICADORES
# =========================
def _norm_lbl(x: object) -> str:
    t = _norm_dato(x).upper().replace(":", "")
    return re.sub(r"\s+", " ", t).strip()


def detectar_indicadores(df: pd.DataFrame, max_rows: int = 60, max_cols: int = 60) -> Dict[str, Any]:
    R = min(max_rows, df.shape[0])
    C = min(max_cols, df.shape[1])
    keys = set(INDICADORES)
    out: Dict[str, Any] = {}

    for r in range(R):
        for c in range(C):
            lbl = _norm_lbl(df.iat[r, c])
            if not lbl:
                continue
            if lbl in keys:
                clave = lbl
            elif lbl.startswith("FR ACTUAL"):
                clave = "FR ACTUAL"
            elif lbl.startswith("FR FINAL"):
                clave = "FR FINAL"
            else:
                continue

            valor = None
            # derecha
            for j in range(c + 1, C):
                v = df.iat[r, j]
                if _non_empty_sem(v):
                    valor = v
                    break
            # izquierda
            if valor is None:
                for j in range(c - 1, -1, -1):
                    v = df.iat[r, j]
                    if _non_empty_sem(v):
                        valor = v
                        break
            out[clave] = valor

    return out


def extraer_tabla_indicadores(df: pd.DataFrame) -> Dict[str, Any]:
    lote = extraer_lote(df)
    fecha_txt = extraer_fecha(df)

    # Intento simple de parsear yyyy/mm/dd desde el texto (opcional)
    fecha = ""
    try:
        t = _norm_dato(fecha_txt)
        m_y = re.search(r"(19|20)\d{2}", t)
        meses = {
            "enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
            "julio":7,"agosto":8,"septiembre":9,"setiembre":9,"octubre":10,
            "noviembre":11,"diciembre":12
        }
        m_d = re.search(r"\b([0-3]?\d)\b", t)
        m_m = next((meses[k] for k in meses if k in t), None)
        if m_y and m_d and m_m:
            fecha = f"{int(m_d.group(1)):02d}/{m_m:02d}/{int(m_y.group(0)):04d}"
    except Exception:
        pass

    d = detectar_indicadores(df)

    cols = ["LOTE", "FECHA_EFECTIVA_TXT", "FECHA.AÑO"] + list(INDICADORES)
    base = {k: "" for k in cols}
    base["LOTE"] = lote
    base["FECHA_EFECTIVA_TXT"] = fecha_txt
    base["FECHA.AÑO"] = fecha

    for k, v in d.items():
        kk = k
        if kk not in INDICADORES:
            if kk.startswith("FR ACTUAL"): kk = "FR ACTUAL"
            elif kk.startswith("FR FINAL"): kk = "FR FINAL"
        if kk in base:
            base[kk] = v

    sub = pd.DataFrame([base], columns=cols)

    num_cols = [c for c in sub.columns if c not in {"LOTE","FECHA_EFECTIVA_TXT","FECHA.AÑO"}]
    sub.loc[:, num_cols] = _numify_df(sub.loc[:, num_cols])

    meta = {"indicadores_detectados": d}
    return {"data": sub, "meta": meta}

# =========================
# Orquestación por archivo (flujo original)
# =========================
def _dbg(msg: str, debug: bool, f=None):
    if debug:
        print(msg)
    if f is not None:
        f.write(msg + "\n")


def procesar_hoja(df: pd.DataFrame, sheet_name: str) -> Tuple[Optional[str], pd.DataFrame]:
    tipo = clasificar_hoja_por_frase(df)
    if not tipo:
        pos_try = detectar_anclas_reservas(df)
        if "categoria" in pos_try and ("limite" in pos_try or "contrato" in pos_try):
            tipo = "RESERVAS"

    if tipo == "RESERVAS":
        pos = detectar_anclas_reservas(df)
        res = completar_tabla_reservas(df, pos)
        sub = res.get("sub", pd.DataFrame())

        # >>> Fallback para tablas sin eje de años (C#13/C#14)
        if sub is None or sub.empty:
            sub = extraer_reservas_sin_anios(df, sheet_name)

        if sub is not None and not sub.empty:
            sub.insert(0, "HOJA", sheet_name)
            sub["ESCENARIO"] = "RESERVAS"
        return tipo, sub
    else:
        return None, pd.DataFrame()


def ejecutar_flujo(input_xlsx: str,
                   out_dir: str = "salidas_rrp",
                   single_file_name: str = "RRyR_unificado",
                   generar_individuales: bool = True,
                   generar_unico: bool = True,
                   debug: bool = False,
                   per_doc_txt: Optional[str] = None) -> List[str]:
    generados: List[str] = []
    if not os.path.exists(input_xlsx):
        raise FileNotFoundError(input_xlsx)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, f"depuracion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    with open(log_path, "w", encoding="utf-8") as flog:
        _dbg(f"[INFO] Archivo: {input_xlsx}", debug, flog)
        try:
            # Si es CSV, normalizamos a DataFrame único simulado como una hoja
            if input_xlsx.lower().endswith(".csv"):
                df = pd.read_csv(input_xlsx, header=None, dtype=object)
                hojas = {"CSV": df}
            else:
                xls = pd.ExcelFile(input_xlsx, engine="openpyxl")
                hojas = {s: xls.parse(sheet_name=s, header=None, dtype=object) for s in xls.sheet_names}
        except Exception as e:
            _dbg(f"[ERROR] No se pudo abrir el documento: {e}", debug, flog)
            return generados

        _dbg(f"[INFO] Hojas detectadas: {list(hojas.keys())}", debug, flog)

        colecta: List[pd.DataFrame] = []
        secciones: List[Tuple[str, str]] = []
        for sheet, df in hojas.items():
            _dbg(f"\n[HOJA] >>> {sheet}", debug, flog)
            tipo, sub = procesar_hoja(df, str(sheet))
            _dbg(f"[INFO] Tipo detectado: {tipo}", debug, flog)
            if not tipo or sub is None or sub.empty:
                plano = _plane(df)
                _dbg(f"[SKIP] Sin match o extractor vacío. Snippet: {plano[:300]}", debug, flog)
                continue

            sub = _flatten_and_make_unique_cols(sub)

            if generar_individuales:
                safe_sheet = re.sub(r"[^\w\-]+", "_", str(sheet)).strip("_")
                base = Path(input_xlsx).stem
                out_path = os.path.join(out_dir, f"{base}__{safe_sheet}__{tipo}.json")
                sub.to_json(out_path, orient='records', indent=2, force_ascii=False)
                generados.append(out_path)
                _dbg(f"[OK] Guardado: {out_path}", debug, flog)

            colecta.append(sub)
            secciones.append((str(sheet), tipo))
            _dbg(f"[INFO] Filas={len(sub)}, NoVacias={int(sub.notna().any(axis=1).sum())}", debug, flog)

        if per_doc_txt and colecta:
            try:
                consolidado_doc = pd.concat(colecta, ignore_index=True, sort=False)
                preferidas = [c for c in ["HOJA", "ESCENARIO", "LOTE", "FECHA_EFECTIVA_TXT", "FECHA.AÑO", "CATEGORIA"] if c in consolidado_doc.columns]
                otras = [c for c in consolidado_doc.columns if c not in preferidas]
                consolidado_doc = consolidado_doc[preferidas + otras]
                Path(per_doc_txt).parent.mkdir(parents=True, exist_ok=True)
                consolidado_doc.to_json(per_doc_txt, orient='records', indent=2, force_ascii=False)
                generados.append(per_doc_txt)
                _dbg(f"[OK] Consolidado por documento: {per_doc_txt}", debug, flog)
            except Exception as e:
                _dbg(f"[ERROR] Falló consolidado por documento: {e}", debug, flog)

    return generados

# =========================
# Indexador recursivo + Registro Global
# =========================
REG_COLS = ["archivo", "procesado", "fecha_procesamiento", "intentos", "estado"]

def _yymmdd_today() -> str:
    return datetime.now().strftime("%Y/%m/%d")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _scan_input_files(root: Path, exts: Tuple[str, ...] = (".xlsx", ".csv")) -> List[Path]:
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(exts):
                files.append(Path(dirpath) / fn)
    return sorted(files)


def _load_or_init_registry(out_dir: Path) -> pd.DataFrame:
    _ensure_dir(out_dir)
    reg_path = out_dir / cfg.REG_NAME
    if reg_path.exists():
        df = pd.read_csv(reg_path)
        for c in REG_COLS:
            if c not in df.columns:
                if c == "fecha_procesamiento":
                    df[c] = ""
                elif c == "procesado":
                    df[c] = False
                elif c == "estado":
                    df[c] = "pendiente"
                elif c == "intentos":
                    df[c] = 0
                elif c == "archivo":
                    df[c] = ""
        return df[REG_COLS]
    else:
        return pd.DataFrame(columns=REG_COLS)


def _save_registry(df: pd.DataFrame, out_dir: Path) -> Path:
    reg_path = out_dir / cfg.REG_NAME
    df.to_csv(reg_path, index=False,encoding='utf-8')
    return reg_path


def _upsert_pending_entry(reg: pd.DataFrame, rel_path: str) -> pd.DataFrame:
    if reg.empty or not (reg["archivo"] == rel_path).any():
        reg = pd.concat([
            reg,
            pd.DataFrame([{
                "archivo": rel_path,
                "procesado": False,
                "fecha_procesamiento": "",
                "intentos": 0,
                "estado": "pendiente",
            }])
        ], ignore_index=True)
    return reg


def _increment_attempt(reg: pd.DataFrame, rel_path: str) -> pd.DataFrame:
    mask = (reg["archivo"] == rel_path)
    reg.loc[mask, "intentos"] = reg.loc[mask, "intentos"].fillna(0).astype(int) + 1
    return reg


def _mark_success(reg: pd.DataFrame, rel_path: str) -> pd.DataFrame:
    mask = (reg["archivo"] == rel_path)
    reg.loc[mask, "procesado"] = True
    reg.loc[mask, "estado"] = "ok"
    reg.loc[mask, "fecha_procesamiento"] = _yymmdd_today()
    return reg


def _mark_error(reg: pd.DataFrame, rel_path: str, set_date: bool = False) -> pd.DataFrame:
    mask = (reg["archivo"] == rel_path)
    reg.loc[mask, "procesado"] = False
    reg.loc[mask, "estado"] = "error"
    if set_date:
        reg.loc[mask, "fecha_procesamiento"] = _yymmdd_today()
    return reg


def _json_output_path(out_dir: Path, rel_path: str) -> Path:
    rel = Path(rel_path)
    dest_dir = out_dir / rel.parent
    dest_name = f"{rel.stem}_pronostico.json"
    dest = dest_dir / dest_name
    _ensure_dir(dest.parent)
    return dest


def _write_json_stub(json_path: Path, src_abs: Path) -> None:
    content = {
        "ORIGEN": str(src_abs),
        "GENERADO": datetime.now().isoformat(timespec='seconds'),
        "ESTADO_INICIAL": "pendiente",
        "Notas": [
            "Este archivo corresponde al documento de entrada.",
            "Actualiza este contenido cuando completes el procesamiento."
        ]
    }
    _ensure_dir(json_path.parent)
    json_path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")


def indexar_entradas_generar_registro(input_dir: str, out_dir: str) -> Path:
    """
    Escanea input_dir recursivamente, crea 1 JSON por documento (.xlsx/.csv)
    y mantiene un único CSV global de registro en out_dir.
    """
    in_root = Path(input_dir).resolve()
    out_root = Path(out_dir).resolve()
    files = _scan_input_files(in_root, exts=(".xlsx", ".csv"))

    reg = _load_or_init_registry(out_root)

    for src in files:
        rel = os.path.relpath(src, start=in_root)
        rel_norm = rel.replace("\\", "/")
        json_path = _json_output_path(out_root, rel_norm)

        key = json_path.name           # <--- usar nombre del JSON
        reg = _upsert_pending_entry(reg, key)
        if not json_path.exists():
            _write_json_stub(json_path, src)

    return _save_registry(reg, out_root)


def procesar_lote(input_dir: str, out_dir: str, procesador: Optional[Callable[[Path, Path], bool]] = None) -> Path:
    """
    Igual que indexar, pero además intenta procesar cada archivo y actualizar estado:
    - intentos += 1
    - si OK: procesado=True, estado="ok", fecha=YYYY/MM/DD
    - si error: procesado=False, estado="error"
    El callback `procesador(src: Path, out_dir: Path) -> bool` debe retornar True/False.
    Si no se provee, por defecto intenta ejecutar_flujo y considera OK si generó alguna salida.
    """
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
        json_path = _json_output_path(out_root, rel_norm)
        if not json_path.exists():
            _write_json_stub(json_path, src)

        reg = _increment_attempt(reg, key)
        try:
            ok: bool
            if procesador is not None:
                ok = bool(procesador(src, out_root))
                pass
            else:
                # Lógica de procesamiento (ejecutar_flujo)
                salidas = ejecutar_flujo(
                    str(src),
                    out_dir=str(out_root),
                    generar_individuales=False,
                    generar_unico=False,
                    debug=False,
                    per_doc_txt=str(json_path)  # Usamos el nombre correcto del argumento aquí
                )
                print(f"--- DEBUG: Retorno de ejecutar_flujo para {src.name} ---")
                print(f"Valor de 'salidas': {salidas}")
                print(f"Resultado Booleano (ok): {bool(salidas)}")
                print("------------------------------------------")
                ok = bool(salidas)
                reg = _mark_success(reg, key) if ok else _mark_error(reg, key)
        except Exception as e: # <-- CAPTURAR LA EXCEPCIÓN
                # AÑADIR ESTAS LÍNEAS DE DEBUG:
                print(f"--- ERROR CRÍTICO al procesar {src.name} ---")
                import traceback
                traceback.print_exc()
                print("------------------------------------------")
                
                # Marcar como error en el registro sin actualizar la fecha de intento
                reg = _mark_error(reg, key, set_date=False)

    return _save_registry(reg, out_root)

# =========================
# CLI
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=(
        "Extracción optimizada (RRyR + Indicadores) con indexador recursivo y registro global."))

    sub = ap.add_subparsers(dest="cmd", required=False)

    # --- Modo original: un solo archivo
    ap_one = sub.add_parser("one", help="Procesar un único documento (.xlsx o .csv)")
    ap_one.add_argument("--input", required=True, help="Ruta del Excel/CSV de entrada")
    ap_one.add_argument("--out_dir", default="salidas_rrp", help="Carpeta de salida (JSON)")
    ap_one.add_argument("--no_individuales", action="store_true", help="No guardar archivos por hoja")
    ap_one.add_argument("--no_unico", action="store_true", help="No generar archivo unificado")
    ap_one.add_argument("--debug", action="store_true", help="Imprimir log detallado en consola")

    # --- Indexar recursivamente (sin procesar)
    ap_idx = sub.add_parser("index", help="Escanear carpetas y crear JSON por documento + registro global")
    ap_idx.add_argument("--input_dir", required=True, help="Carpeta raíz de entrada (recursiva)")
    ap_idx.add_argument("--out_dir", required=True, help="Carpeta de salida")

    # --- Procesar recursivamente (con actualización de registro)
    ap_proc = sub.add_parser("process", help="Procesar recursivamente y actualizar registro (ok/error)")
    ap_proc.add_argument("--input_dir", required=True, help="Carpeta raíz de entrada (recursiva)")
    ap_proc.add_argument("--out_dir", required=True, help="Carpeta de salida")

    args = ap.parse_args()

    if args.cmd == "one" or (args.cmd is None and hasattr(args, "input")):
        files = ejecutar_flujo(
            input_xlsx=args.input,
            out_dir=args.out_dir,
            generar_individuales=not getattr(args, "no_individuales", False),
            generar_unico=not getattr(args, "no_unico", False),
            debug=getattr(args, "debug", False),
        )
        if files:
            print("Generados:")
            for f in files:
                print(" -", f)
        else:
            print("No se generaron archivos.")

    elif args.cmd == "index":
        reg = indexar_entradas_generar_registro(args.input_dir, args.out_dir)
        print(f"[OK] Registro global en: {reg}")

    elif args.cmd == "process":
        reg = procesar_lote(args.input_dir, args.out_dir)
        print(f"[OK] Registro global actualizado en: {reg}")

    else:
        # Ayuda breve si no se pasa comando
        ap.print_help()
