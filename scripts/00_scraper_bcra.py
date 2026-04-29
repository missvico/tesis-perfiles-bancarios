"""
00_scraper_bcra.py
==================
Scraper del portal de Entidades Financieras del BCRA.
Descarga estados contables, situación de deudores e información de estructura
para cada banco, y guarda todo en archivos CSV + una base de datos SQLite.

Uso:
    python 00_scraper_bcra.py

Requiere:
    pip install requests beautifulsoup4 pandas
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import sqlite3
import time
import os
import logging
from datetime import datetime
from io import StringIO

# ── Configuración ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DELAY = 1.5          # segundos entre requests (ser respetuosos con el BCRA)
TIMEOUT = 20         # timeout por request
OUTPUT_DIR = "data/raw/bcra_scraping"
DB_PATH = "data/processed/bcra_bancos.db"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ── Listado de bancos (solo códigos 00XXX, excluidas financieras 44XXX/65XXX) ─

BANCOS = {
    "00340": "BACS BANCO DE CREDITO Y SECURITIZAC",
    "00017": "BANCO BBVA ARGENTINA S.A.",
    "00426": "BANCO BICA S.A.",
    "00331": "BANCO CETELEM ARGENTINA S.A.",
    "00319": "BANCO CMF S.A.",
    "00431": "BANCO COINAG S.A.",
    "00389": "BANCO COLUMBIA S.A.",
    "00299": "BANCO COMAFI SOCIEDAD ANONIMA",
    "00191": "BANCO CREDICOOP COOPERATIVO LIMITAD",
    "00432": "BANCO DE COMERCIO S.A.",
    "00094": "BANCO DE CORRIENTES S.A.",
    "00315": "BANCO DE FORMOSA S.A.",
    "00007": "BANCO DE GALICIA Y BUENOS AIRES S.A",
    "00300": "BANCO DE INVERSION Y COMERCIO EXTER",
    "00029": "BANCO DE LA CIUDAD DE BUENOS AIRES",
    "00011": "BANCO DE LA NACION ARGENTINA",
    "00093": "BANCO DE LA PAMPA SA",
    "00014": "BANCO DE LA PROVINCIA DE BUENOS AIR",
    "00020": "BANCO DE LA PROVINCIA DE CORDOBA S.",
    "00269": "BANCO DE LA REPUBLICA ORIENTAL DEL",
    "00045": "BANCO DE SAN JUAN S.A.",
    "00086": "BANCO DE SANTA CRUZ S.A.",
    "00321": "BANCO DE SANTIAGO DEL ESTERO S.A.",
    "00332": "BANCO DE SERVICIOS FINANCIEROS S.A.",
    "00338": "BANCO DE SERVICIOS Y TRANSACCIONES",
    "00198": "BANCO DE VALORES S.A.",
    "00083": "BANCO DEL CHUBUT S.A.",
    "00310": "BANCO DEL SOL S.A.",
    "00448": "BANCO DINO S.A.",
    "00044": "BANCO HIPOTECARIO S.A.",
    "00322": "BANCO INDUSTRIAL S.A.",
    "00305": "BANCO JULIO SOCIEDAD ANONIMA",
    "00285": "BANCO MACRO S.A.",
    "00254": "BANCO MARIVA S.A.",
    "00341": "BANCO MASVENTAS S.A.",
    "00281": "BANCO MERIDIAN S.A.",
    "00065": "BANCO MUNICIPAL DE ROSARIO",
    "00034": "BANCO PATAGONIA S.A.",
    "00301": "BANCO PIANO S.A.",
    "00268": "BANCO PROVINCIA DE TIERRA DEL FUEGO",
    "00097": "BANCO PROVINCIA DEL NEUQUÉN SOCIEDA",
    "00309": "BANCO RIOJA SOCIEDAD ANONIMA UNIPER",
    "00247": "BANCO ROELA S.A.",
    "00277": "BANCO SAENZ S.A.",
    "00072": "BANCO SANTANDER ARGENTINA S.A.",
    "00435": "BANCO SUCREDITO REGIONAL S.A.U.",
    "00027": "BANCO SUPERVIELLE S.A.",
    "00312": "BANCO VOII S.A.",
    "00131": "BANK OF CHINA LIMITED SUCURSAL BUEN",
    "00147": "BIBANK S.A.",
    "00266": "BNP PARIBAS",
    "00143": "BRUBANK S.A.U.",
    "00016": "CITIBANK N.A.",
    "00015": "INDUSTRIAL AND COMMERCIAL BANK OF C",
    "00165": "JPMORGAN CHASE BANK, NATIONAL ASSOC",
    "00384": "UALA BANK S.A.U.",
}

# ── Clasificación manual por tipo (público / privado nacional / extranjero) ──
# Fuente: conocimiento del sistema financiero argentino.
# Verificar y actualizar si hubiera cambios recientes.

TIPO_ENTIDAD = {
    "00011": "publico",           # Banco Nación
    "00014": "publico",           # Provincia Buenos Aires
    "00020": "publico",           # Provincia Córdoba
    "00029": "publico",           # Ciudad de Buenos Aires
    "00044": "publico",           # Hipotecario
    "00045": "publico",           # San Juan
    "00065": "publico",           # Municipal de Rosario
    "00083": "publico",           # Chubut
    "00086": "publico",           # Santa Cruz
    "00093": "publico",           # La Pampa
    "00094": "publico",           # Corrientes
    "00097": "publico",           # Neuquén
    "00268": "publico",           # Tierra del Fuego
    "00300": "publico",           # BICE
    "00309": "publico",           # Rioja
    "00311": "publico",           # Nuevo Banco del Chaco  ← faltaba en BANCOS, agregar si aparece
    "00315": "publico",           # Formosa
    "00321": "publico",           # Santiago del Estero
    "00330": "publico",           # Nuevo Banco de Santa Fe
    "00386": "publico",           # Nuevo Banco de Entre Ríos
    "00007": "privado_nacional",  # Galicia
    "00017": "extranjero",        # BBVA
    "00027": "privado_nacional",  # Supervielle
    "00034": "privado_nacional",  # Patagonia
    "00072": "extranjero",        # Santander
    "00131": "extranjero",        # Bank of China
    "00143": "privado_nacional",  # Brubank
    "00147": "privado_nacional",  # Bibank
    "00165": "extranjero",        # JPMorgan
    "00191": "privado_nacional",  # Credicoop
    "00198": "privado_nacional",  # Banco de Valores
    "00247": "privado_nacional",  # Roela
    "00254": "privado_nacional",  # Mariva
    "00266": "extranjero",        # BNP Paribas
    "00269": "extranjero",        # BROU
    "00277": "privado_nacional",  # Saenz
    "00281": "privado_nacional",  # Meridian
    "00285": "privado_nacional",  # Macro
    "00299": "privado_nacional",  # Comafi
    "00301": "privado_nacional",  # Piano
    "00305": "privado_nacional",  # Julio
    "00310": "privado_nacional",  # Del Sol
    "00312": "privado_nacional",  # Voii
    "00319": "privado_nacional",  # CMF
    "00322": "privado_nacional",  # Industrial
    "00331": "extranjero",        # Cetelem
    "00332": "privado_nacional",  # BSF
    "00338": "privado_nacional",  # BST
    "00339": "extranjero",        # RCI Banque
    "00340": "privado_nacional",  # BACS
    "00341": "privado_nacional",  # Masventas
    "00384": "privado_nacional",  # Uala Bank
    "00389": "privado_nacional",  # Columbia
    "00426": "privado_nacional",  # Bica
    "00431": "privado_nacional",  # Coinag
    "00432": "privado_nacional",  # Banco de Comercio
    "00435": "privado_nacional",  # Sucredito
    "00448": "privado_nacional",  # Dino
    "00015": "extranjero",        # ICBC
    "00016": "extranjero",        # Citibank
}

# ── Funciones de scraping ─────────────────────────────────────────────────────

def get_html(url: str) -> BeautifulSoup | None:
    """Descarga una URL y devuelve el objeto BeautifulSoup."""
    try:
        r = requests.get(url, verify=False, timeout=TIMEOUT)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        log.warning(f"Error al descargar {url}: {e}")
        return None


def parse_tablas(soup: BeautifulSoup, codigo: str, nombre: str, fuente: str) -> pd.DataFrame | None:
    """
    Extrae todas las tablas de datos de una página del BCRA.
    Convierte el formato wide (columnas = períodos) a long (una fila por variable × período).
    """
    try:
        # pandas 3.0 exige un buffer tipo file (StringIO) en lugar de un string crudo.
        # decimal="," y thousands="." porque los números del BCRA vienen en formato español
        # (ej. "2.814.492,63" y "96,73"). Sin esto, pandas interpreta "96,73" como "9673".
        tablas = pd.read_html(StringIO(str(soup)), decimal=",", thousands=".")
    except Exception as e:
        log.debug(f"read_html falló para {fuente} de {codigo}: {e}")
        return None

    tablas_limpias = []
    for t in tablas:
        if t.shape[1] < 2:
            continue
        # Algunas tablas vienen con MultiIndex en columnas (ej. situacion_deudores).
        # Aplanamos tomando el último nivel no vacío (donde está el período real).
        if isinstance(t.columns, pd.MultiIndex):
            nuevas_cols = []
            for col in t.columns:
                niveles = [str(x) for x in col if str(x) not in ("", "nan")]
                nuevas_cols.append(niveles[-1] if niveles else "col")
            t = t.copy()
            t.columns = nuevas_cols
        # Descartar la tabla "INFORMES" (lista de códigos de reporte, no tiene períodos)
        primera_col = str(t.columns[0]).upper()
        if primera_col.startswith("INFORMES") or primera_col.startswith("TIPO"):
            continue
        tablas_limpias.append(t)
    tablas = tablas_limpias

    if not tablas:
        return None

    frames = []
    for tabla in tablas:
        # La primera columna suele ser el nombre de la variable
        col_variable = tabla.columns[0]
        cols_periodos = tabla.columns[1:]

        # Convertir de wide a long
        df_long = tabla.melt(
            id_vars=[col_variable],
            value_vars=cols_periodos,
            var_name="periodo",
            value_name="valor"
        )
        df_long = df_long.rename(columns={col_variable: "variable"})
        frames.append(df_long)

    if not frames:
        return None

    resultado = pd.concat(frames, ignore_index=True)
    resultado.insert(0, "codigo_bcra", codigo)
    resultado.insert(1, "nombre_banco", nombre)
    resultado.insert(2, "tipo_entidad", TIPO_ENTIDAD.get(codigo, "no_clasificado"))
    resultado.insert(3, "fuente", fuente)
    resultado["scraping_fecha"] = datetime.today().strftime("%Y-%m-%d")

    # `pd.read_html(..., decimal=",", thousands=".")` ya parseó los números en formato español,
    # así que sólo forzamos el cast a numérico para las celdas que pandas haya dejado como texto.
    resultado["valor"] = pd.to_numeric(resultado["valor"], errors="coerce")

    return resultado


# ── Scraping principal ────────────────────────────────────────────────────────

ENDPOINTS_HTML = {
    "estados_contables":    "https://www.bcra.gob.ar/entidades-financieras-estados-contables/?bco={codigo}",
    "situacion_deudores":   "https://www.bcra.gob.ar/entidades-financieras-situacion-deudores/?bco={codigo}",
    "informacion_estructura": "https://www.bcra.gob.ar/entidades-financieras-informacion-estructura/?bco={codigo}",
}

API_INDICADORES = "https://www.bcra.gob.ar/api-indicadores-economicos.php?action=indicadores&bco={codigo}"

# Lista completa de fuentes (para crear carpetas y consolidados)
ENDPOINTS = {**ENDPOINTS_HTML, "indicadores_economicos": API_INDICADORES}


def scrape_indicadores_json(codigo: str, nombre: str) -> pd.DataFrame | None:
    """Consume la API JSON interna del BCRA para indicadores económicos.

    La página pública se renderiza con JS y no expone tablas vía HTML estático,
    pero el front-end golpea este endpoint JSON. Devuelve el mismo schema `long`
    que el resto del scraper.
    """
    url = API_INDICADORES.format(codigo=codigo)
    try:
        r = requests.get(url, verify=False, timeout=TIMEOUT)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        log.warning(f"  [{codigo}] indicadores_economicos API: {e}")
        return None

    if not payload.get("success"):
        return None

    # Mapeo col1..col5 → nombre de período real
    col_to_periodo = payload.get("columnas", {})
    if not col_to_periodo:
        return None

    filas = []
    for seccion, items in payload.get("secciones", {}).items():
        for item in items:
            titulo = item.get("in_titulo", "").strip()
            for col_key, periodo in col_to_periodo.items():
                # col_key = "col1" → valor en item["in_c1"]
                n = col_key.replace("col", "")
                valor = item.get(f"in_c{n}")
                filas.append({
                    "variable": f"{seccion}: {titulo}",
                    "periodo": periodo,
                    "valor": valor,
                })

    if not filas:
        return None

    df = pd.DataFrame(filas)
    df.insert(0, "codigo_bcra", codigo)
    df.insert(1, "nombre_banco", nombre)
    df.insert(2, "tipo_entidad", TIPO_ENTIDAD.get(codigo, "no_clasificado"))
    df.insert(3, "fuente", "indicadores_economicos")
    df["scraping_fecha"] = datetime.today().strftime("%Y-%m-%d")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df


def scrape_banco(codigo: str, nombre: str) -> dict[str, pd.DataFrame]:
    """Scrapea todos los endpoints para un banco. Devuelve dict fuente → DataFrame."""
    resultados = {}

    # Endpoints HTML (estados contables, situación de deudores, información de estructura)
    for fuente, url_template in ENDPOINTS_HTML.items():
        url = url_template.format(codigo=codigo)
        soup = get_html(url)
        if soup is None:
            log.warning(f"  [{codigo}] Sin datos para {fuente}")
            continue

        df = parse_tablas(soup, codigo, nombre, fuente)
        if df is not None and not df.empty:
            resultados[fuente] = df
            log.info(f"  [{codigo}] {fuente}: {len(df)} filas")
        else:
            log.warning(f"  [{codigo}] {fuente}: tabla vacía o no encontrada")

        time.sleep(DELAY)

    # Endpoint JSON (indicadores económicos)
    df_ind = scrape_indicadores_json(codigo, nombre)
    if df_ind is not None and not df_ind.empty:
        resultados["indicadores_economicos"] = df_ind
        log.info(f"  [{codigo}] indicadores_economicos: {len(df_ind)} filas")
    else:
        log.warning(f"  [{codigo}] indicadores_economicos: sin datos")
    time.sleep(DELAY)

    return resultados


def guardar_csv(df: pd.DataFrame, fuente: str, codigo: str):
    """Guarda el DataFrame como CSV individual por banco y fuente."""
    path = os.path.join(OUTPUT_DIR, fuente)
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path, f"{codigo}.csv"), index=False)


def guardar_sqlite(df: pd.DataFrame, tabla: str, conn: sqlite3.Connection):
    """Agrega el DataFrame a una tabla SQLite (crea la tabla si no existe)."""
    df.to_sql(tabla, conn, if_exists="append", index=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info(f"Iniciando scraping de {len(BANCOS)} bancos...")
    conn = sqlite3.connect(DB_PATH)

    todos = {fuente: [] for fuente in ENDPOINTS}

    for i, (codigo, nombre) in enumerate(BANCOS.items(), 1):
        log.info(f"[{i}/{len(BANCOS)}] {nombre} ({codigo})")
        resultados = scrape_banco(codigo, nombre)

        for fuente, df in resultados.items():
            guardar_csv(df, fuente, codigo)
            guardar_sqlite(df, fuente, conn)
            todos[fuente].append(df)

    # Guardar CSVs consolidados (un archivo por fuente con todos los bancos)
    for fuente, frames in todos.items():
        if frames:
            df_total = pd.concat(frames, ignore_index=True)
            path_total = os.path.join("data/processed", f"{fuente}_todos.csv")
            df_total.to_csv(path_total, index=False)
            log.info(f"Consolidado guardado: {path_total} ({len(df_total)} filas)")

    conn.close()
    log.info(f"Listo. Base de datos en: {DB_PATH}")
    log.info("Tablas creadas en SQLite:")
    for fuente in ENDPOINTS:
        log.info(f"  - {fuente}")


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    main()
