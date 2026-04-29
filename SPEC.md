# SPEC.md — Tesis de Maestría: Perfiles Bancarios en Argentina
**Autora:** Victoria Di Liscia  
**Programa:** Maestría en Explotación de Datos y Descubrimiento del Conocimiento — FCEN, UBA  
**Entrega actual:** Entrega II — Análisis Exploratorio de Datos (EDA)  
**Última actualización:** Abril 2026

---

## 1. Pregunta de investigación

¿Existen perfiles diferenciados de bancos en Argentina según balance, rentabilidad y riesgo? ¿Qué variables determinan cada perfil?

---

## 2. Objetivo de esta entrega (EDA)

Cargar, explorar y documentar los datasets para:
- Entender la estructura y calidad de cada fuente de datos.
- Identificar variables relevantes para el clustering posterior.
- Detectar problemas de calidad (missings, outliers, tipos incorrectos).
- Producir visualizaciones descriptivas alineadas con los objetivos del proyecto.

**No se aplican técnicas de data mining en esta entrega.**

---

## 3. Decisiones de diseño confirmadas

| Dimensión | Decisión |
|-----------|----------|
| Universo | **Solo bancos** — excluidas financieras (códigos 44XXX, 65XXX). Total: 56 entidades. |
| Clasificación de entidades | Ya resuelta: codificada en `TIPO_ENTIDAD` dentro del scraper |
| Ventana temporal | Datos de corte reciente (2025-2026). A confirmar período exacto en el EDA según el scraper |
| Unidad de análisis para clustering | Corte transversal: una fila por banco |
| Gráficos | Solo visualizar en el notebook (sin exportar) |
| Formato de trabajo | Jupyter Notebook (.ipynb) |
| Base de datos | SQLite (`data/processed/bcra_bancos.db`) + CSVs consolidados |

---

## 4. Fuentes de datos

### 4.1 Portal de Entidades Financieras del BCRA — scraping web (FUENTE PRINCIPAL)

**Script:** `scripts/00_scraper_bcra.py` ← **YA EXISTE, correrlo primero**

```bash
pip install requests beautifulsoup4 pandas
python scripts/00_scraper_bcra.py
```

Scrapea 4 endpoints para cada uno de los 56 bancos:

| Endpoint | Tabla SQLite | Contenido | Unidad monetaria |
|----------|-------------|-----------|-----------------|
| `estados_contables` | `estados_contables` | Balance: activo, pasivo, PN, resultados | Miles de $ |
| `situacion_deudores` | `situacion_deudores` | Cartera por situación crediticia y tipo | Ratios en % |
| `indicadores_economicos` | `indicadores_economicos` | ROE, ROA, liquidez, solvencia | ⚠️ Puede salir vacío (carga con JS) |
| `informacion_estructura` | `informacion_estructura` | Cuentas, tarjetas, personal, sucursales | Cantidades |

**Períodos disponibles por endpoint (confirmados):**
- Estados contables: Dic-2023, Dic-2024, Nov-2025, Dic-2025, Ene-2026
- Situación deudores: Dic-2023, Dic-2024, Oct-2025, Nov-2025, Dic-2025
- Información estructura: Dic-2023, Dic-2024, Jun-2025, Set-2025, Dic-2025

**Formato del output (long):**
```
codigo_bcra | nombre_banco | tipo_entidad | fuente | variable | periodo | valor | scraping_fecha
00007       | BANCO GALICIA | privado_nac | estados_contables | ACTIVO | Dic-2024 | 21692306233 | 2026-04-19
```

**Clasificación de entidades (ya codificada en el scraper):**
- `publico`: Nación, Provincia BsAs, CABA, Córdoba, Hipotecario, San Juan, Chubut, Santa Cruz, La Pampa, Corrientes, Neuquén, Tierra del Fuego, BICE, Rioja, Formosa, Santiago del Estero
- `privado_nacional`: Galicia, Macro, Supervielle, Credicoop, Patagonia, Brubank, Bibank, Comafi, Piano, BST, BSF, BACS, Columbia, Masventas, Uala, Meridian, Mariva, Roela, Saenz, Voii, CMF, Industrial, Julio, Del Sol, y otros
- `extranjero`: BBVA, Santander, ICBC, Citibank, BNP Paribas, JPMorgan, Bank of China, BROU, Cetelem, RCI Banque

---

### 4.2 Régimen Informativo de Transparencia — Sección 36 (FUENTE ADICIONAL)

**Fuente normativa:** Comunicación "A" 8188 del BCRA (Sección 36)  
**Ubicación en el proyecto:** `data/raw/transparencia/`

#### Estructura real de los archivos (verificada)

Todos los archivos tienen:
- **Separador:** `;`
- **Encoding:** `latin-1` (ANSI-1252) — obligatorio al leer
- **Decimales:** `,` como separador (ej: `25,00` = 25.00%) — convertir a `.` al procesar
- **Headers en español** con nombres descriptivos en la primera fila
- **Columnas 1 y 2 siempre presentes:** `Código de Entidad` y `Descripción de Entidad` ← clave para el join con el scraper
- **Columna 3:** `Fecha de Información` — snapshot del momento en que el banco actualizó ese producto

#### Archivos disponibles y sus columnas

**HIPOTECA.CSV** — 373 filas, 29 entidades (25 bancos del scraper)
```
Código de Entidad | Descripción de Entidad | Fecha de Información |
Nombre completo del Préstamo Hipotecario | Nombre corto | Denominación |
Monto máximo otorgable | Plazo máximo otorgable | Ingreso mínimo mensual |
Antigüedad laboral mínima (meses) | Edad máxima | Relación cuota/ingreso (%) |
Relación monto/tasación (%) | Destino de los fondos | Beneficiarios |
Cargo máximo por cancelación anticipada | Tasa efectiva anual máxima |
Tipo de Tasa | Costo financiero efectivo total máximo |
Cuota inicial a plazo máximo cada $100.000 | Territorio | Más información
```

**PERSONALES.CSV** — 2537 filas, 56 entidades (43 bancos del scraper)
```
Código de Entidad | Descripción de Entidad | Fecha de Información |
Nombre completo del Préstamo Personal | Nombre corto | Denominación |
Monto máximo otorgable | Monto mínimo otorgable | Plazo máximo otorgable |
Ingreso mínimo mensual | Antigüedad laboral mínima (meses) | Edad máxima |
Relación cuota/ingreso (%) | Beneficiario | Cargo máximo por cancelación anticipada |
Tasa efectiva anual máxima | Tipo de Tasa | Costo financiero efectivo total máximo |
Cuota inicial a plazo máximo cada $10.000 | Territorio | Más información
```

**PRENDARIOS.CSV** — 571 filas, 47 entidades (19 bancos del scraper)
```
Código de Entidad | Descripción de Entidad | Fecha de Información |
Nombre completo del Préstamo Prendario | Nombre corto | Denominación |
Monto máximo otorgable | Monto mínimo otorgable | Plazo máximo otorgable |
Ingreso mínimo mensual | Antigüedad laboral mínima (meses) | Edad máxima |
Relación cuota/ingreso (%) | Relación monto/tasación (%) | Destino de los fondos |
Beneficiario | Cargo máximo por cancelación anticipada |
Tasa efectiva anual máxima | Tipo de Tasa | Costo financiero efectivo total máximo |
Cuota inicial a plazo máximo cada $10.000 | Territorio | Más información
```

**PFIJO.CSV** — 901 filas, 73 entidades (50 bancos del scraper)
```
Código de Entidad | Descripción de Entidad | Fecha de Información |
Nombre completo del plazo fijo | Nombre corto | Denominación |
Monto mínimo a invertir | Plazo mínimo a invertir | Canal de constitución |
Tasa efectiva anual mínima | Territorio | Más información
```

**PAQUETE.CSV** — 1088 filas, 33 entidades (28 bancos del scraper)
```
Código de Entidad | Descripción de Entidad | Fecha de Información |
Nombre completo del Paquete de Productos | Nombre corto |
Comisión máxima por servicio de mantenimiento de paquete |
Ingreso mínimo mensual | Antigüedad laboral mínima (meses) | Edad máxima |
Beneficiarios | Segmento | Productos que integran el paquete | Territorio | Más información
```

**TARJETAS.CSV** — 546 filas, 139 entidades (41 bancos del scraper)
```
Código de Entidad | Descripción de Entidad | Fecha de Información |
Nombre completo de la tarjeta de crédito | Nombre corto |
Comisión máxima por administración y mantenimiento de la cuenta |
Comisión máxima por servicio de renovación |
Tasa efectiva anual máxima de interés compensatorio por financiación de saldos |
Tasa efectiva anual máxima de interés por adelanto en efectivo |
Ingreso mínimo mensual | Antigüedad laboral mínima (meses) | Edad máxima |
Segmento | Territorio | Más información
```

**CAJADEAHORROS.CSV** — 62 filas, 61 entidades (46 bancos del scraper)
```
Código de Entidad | Descripción de Entidad | Fecha de Información |
¿Estableció el proceso simplificado de debida diligencia de identificación del cliente para su apertura?
```

#### Cómo leer en Python

```python
import pandas as pd

def leer_transparencia(path):
    df = pd.read_csv(path, sep=";", encoding="latin-1")
    # Normalizar columna clave de join
    df = df.rename(columns={df.columns[0]: "codigo_entidad",
                             df.columns[1]: "nombre_entidad",
                             df.columns[2]: "fecha_info"})
    df["codigo_entidad"] = df["codigo_entidad"].astype(str).str.zfill(5)  # → "00007"
    # Convertir decimales con coma a punto en columnas numéricas
    for col in df.select_dtypes("object").columns:
        convertida = df[col].str.replace(",", ".", regex=False)
        numerica = pd.to_numeric(convertida, errors="coerce")
        if numerica.notna().sum() > len(df) * 0.3:  # si >30% son números, convertir
            df[col] = numerica
    return df

hipoteca   = leer_transparencia("data/raw/transparencia/HIPOTECA.CSV")
personales = leer_transparencia("data/raw/transparencia/PERSONALES.CSV")
prendarios = leer_transparencia("data/raw/transparencia/PRENDARIOS.CSV")
pfijo      = leer_transparencia("data/raw/transparencia/PFIJO.CSV")
paquete    = leer_transparencia("data/raw/transparencia/PAQUETE.CSV")
tarjetas   = leer_transparencia("data/raw/transparencia/TARJETAS.CSV")
cajas      = leer_transparencia("data/raw/transparencia/CAJADEAHORROS.CSV")
```

#### Cómo agregar a nivel banco para el join

Cada archivo tiene múltiples filas por banco (un producto por fila). Hay que agregar antes del join:

```python
def agregar_por_banco(df_hip, df_per, df_pren, df_pf, df_paq, df_tar, df_ca):

    # HIPOTECA: ¿ofrece? ¿en UVA? TEA máxima, monto máximo, plazo máximo
    hip = df_hip.groupby("codigo_entidad").agg(
        ofrece_hipoteca=("Tasa efectiva anual máxima", lambda x: 1),
        hipoteca_uva=("Denominación", lambda x: int("UVA" in x.values)),
        hipoteca_tea_max=("Tasa efectiva anual máxima", "min"),   # mejor tasa disponible
        hipoteca_monto_max=("Monto máximo otorgable del préstamo", "max"),
        hipoteca_plazo_max=("Plazo máximo otorgable", "max"),
    ).reset_index().assign(ofrece_hipoteca=1)

    # PERSONALES: ¿ofrece? TEA máxima, monto máximo
    per = df_per.groupby("codigo_entidad").agg(
        ofrece_personal=("Tasa efectiva anual máxima", lambda x: 1),
        personal_tea_max=("Tasa efectiva anual máxima", "min"),
        personal_monto_max=("Monto máximo otorgable", "max"),
        personal_plazo_max=("Plazo máximo otorgable", "max"),
    ).reset_index().assign(ofrece_personal=1)

    # PRENDARIOS: ¿ofrece?
    pren = df_pren.groupby("codigo_entidad").agg(
        ofrece_prendario=("Tasa efectiva anual máxima", lambda x: 1),
        prendario_tea_max=("Tasa efectiva anual máxima", "min"),
    ).reset_index().assign(ofrece_prendario=1)

    # PFIJO: ¿ofrece? ¿en UVA? TEA mínima, monto mínimo
    pf = df_pf.groupby("codigo_entidad").agg(
        ofrece_pfijo=("Tasa efectiva anual mínima", lambda x: 1),
        pfijo_uva=("Denominación", lambda x: int("UVA" in x.values)),
        pfijo_tea_min=("Tasa efectiva anual mínima", "max"),      # mejor tasa para el cliente
        pfijo_monto_min=("Monto mínimo a invertir", "min"),
    ).reset_index().assign(ofrece_pfijo=1)

    # PAQUETE: ¿ofrece? ¿ofrece premium? comisión promedio de mantenimiento
    paq = df_paq.groupby("codigo_entidad").agg(
        ofrece_paquete=("Comisión máxima por servicio de mantenimiento de paquete", lambda x: 1),
        paquete_tiene_premium=("Segmento", lambda x: int(
            x.str.lower().str.contains("platinum|gold|black|signature", na=False).any()
        )),
        paquete_comision_min=("Comisión máxima por servicio de mantenimiento de paquete", "min"),
    ).reset_index().assign(ofrece_paquete=1)

    # TARJETAS: ¿ofrece? TEA financiación, ¿tiene premium?
    tar = df_tar.groupby("codigo_entidad").agg(
        ofrece_tarjeta=("Tasa efectiva anual máxima de interés compensatorio por financiación de saldos", lambda x: 1),
        tarjeta_tea_financiacion=("Tasa efectiva anual máxima de interés compensatorio por financiación de saldos", "min"),
        tarjeta_tiene_premium=("Segmento", lambda x: int(
            x.str.lower().str.contains("platinum|gold|black|signature", na=False).any()
        )),
        tarjeta_comision_admin_min=("Comisión máxima por administración y mantenimiento de la cuenta", "min"),
    ).reset_index().assign(ofrece_tarjeta=1)

    # CAJADEAHORROS: ¿tiene apertura simplificada?
    ca = df_ca.groupby("codigo_entidad").agg(
        caja_apertura_simplificada=("¿Estableció el proceso simplificado de debida diligencia de identificación del cliente para su apertura?",
                                    lambda x: int("SI" in x.str.upper().values))
    ).reset_index()

    # Join progresivo
    from functools import reduce
    dfs = [hip, per, pren, pf, paq, tar, ca]
    transparencia = reduce(lambda l, r: l.merge(r, on="codigo_entidad", how="outer"), dfs)

    # Rellenar NaN en columnas binarias con 0 (banco no ofrece ese producto)
    cols_binarias = [c for c in transparencia.columns if c.startswith("ofrece_") or
                     c.endswith("_uva") or c.endswith("_premium") or c.endswith("_simplificada")]
    transparencia[cols_binarias] = transparencia[cols_binarias].fillna(0).astype(int)

    return transparencia
```

#### Nota sobre cobertura

No todos los bancos reportan en todos los archivos. El EDA debe documentar:
- Bancos del scraper que **no aparecen** en ningún archivo de transparencia
- Bancos que ofrecen productos muy limitados (ej: solo plazo fijo, sin préstamos)
- Esas ausencias son información en sí misma para el clustering (perfil de banco especializado vs. universal)

---

## 5. Estructura de carpetas del proyecto

```
tesis-perfiles-bancarios/
│
├── SPEC.md
├── README.md
│
├── scripts/
│   └── 00_scraper_bcra.py              ← ⚠️ YA EXISTE — correr primero
│
├── notebooks/
│   ├── 01_ingesta_y_calidad.ipynb      ← carga, limpieza, joins entre fuentes
│   ├── 02_eda.ipynb                    ← EDA principal (esta entrega)
│   └── 03_clustering.ipynb             ← (entregas futuras)
│
├── data/
│   ├── raw/
│   │   ├── bcra_scraping/              ← output del scraper (CSVs por banco)
│   │   │   ├── estados_contables/
│   │   │   ├── situacion_deudores/
│   │   │   ├── indicadores_economicos/
│   │   │   └── informacion_estructura/
│   │   └── transparencia/              ← archivos del Régimen Sección 36
│   │       ├── HIPOTECA.CSV
│   │       ├── PERSONALES.CSV
│   │       ├── PRENDARIOS.CSV
│   │       ├── PFIJO.CSV
│   │       ├── PAQUETE.CSV
│   │       ├── TARJETAS.CSV
│   │       └── CAJADEAHORROS.CSV
│   └── processed/
│       ├── bcra_bancos.db              ← SQLite con 4 tablas (output del scraper)
│       ├── estados_contables_todos.csv
│       ├── situacion_deudores_todos.csv
│       ├── indicadores_economicos_todos.csv
│       ├── informacion_estructura_todos.csv
│       ├── transparencia_agregada.csv  ← una fila por banco, variables de oferta
│       └── dataset_master.csv          ← dataset final integrado (una fila por banco)
│
└── requirements.txt
```

---

## 6. Entorno Python

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
requests>=2.31
beautifulsoup4>=4.12
scipy>=1.11
scikit-learn>=1.3
jupyter
ipykernel
missingno
sqlalchemy>=2.0
```

---

## 7. Notebook 01 — Ingesta y calidad

### 7.1 Carga del scraper desde SQLite

```python
import sqlite3, pandas as pd

conn = sqlite3.connect("data/processed/bcra_bancos.db")
ec = pd.read_sql("SELECT * FROM estados_contables", conn)
sd = pd.read_sql("SELECT * FROM situacion_deudores", conn)
es = pd.read_sql("SELECT * FROM informacion_estructura", conn)
conn.close()

# Pivotear de long a wide para un período concreto
def pivotear(df, periodo):
    return (df[df["periodo"] == periodo]
              .pivot_table(index=["codigo_bcra", "nombre_banco", "tipo_entidad"],
                           columns="variable", values="valor", aggfunc="first")
              .reset_index())

ec_wide = pivotear(ec, "Dic-2024")   # ajustar período según disponibilidad real
sd_wide = pivotear(sd, "Dic-2024")
es_wide = pivotear(es, "Dic-2024")
```

### 7.2 Carga del Régimen de Transparencia

Ver código completo en Sección 4.2. Pasos:
1. Leer cada CSV con `encoding="latin-1"`, `sep=";"`
2. Renombrar columnas 0-2 a `codigo_entidad`, `nombre_entidad`, `fecha_info`
3. Normalizar `codigo_entidad` a 5 dígitos con `zfill(5)`
4. Convertir campos numéricos (coma → punto)
5. Agregar por banco con `agregar_por_banco()`

### 7.3 Join final — dataset_master

```python
# Clave de join: codigo_bcra (scraper) ↔ codigo_entidad (transparencia)
# Ambos son el mismo código BCRA en formato string de 5 dígitos

dataset_master = (
    ec_wide
    .rename(columns={"codigo_bcra": "codigo_entidad"})
    .merge(sd_wide.rename(columns={"codigo_bcra": "codigo_entidad"}),
           on=["codigo_entidad", "nombre_banco", "tipo_entidad"], how="left")
    .merge(es_wide.rename(columns={"codigo_bcra": "codigo_entidad"}),
           on=["codigo_entidad", "nombre_banco", "tipo_entidad"], how="left")
    .merge(transparencia_agregada, on="codigo_entidad", how="left")
)

dataset_master.to_csv("data/processed/dataset_master.csv", index=False)
print(f"Dataset final: {dataset_master.shape[0]} bancos × {dataset_master.shape[1]} variables")
```

### 7.4 Tareas de calidad de datos

- [ ] % de missings por variable y por banco en cada fuente
- [ ] `missingno.matrix()` para el dataset_master final
- [ ] Verificar tipos de datos correctos (especialmente decimales con coma)
- [ ] Documentar bancos del scraper que no aparecen en transparencia (ausencia = información)
- [ ] Detectar outliers en TEA, comisiones, activos — documentar, no eliminar
- [ ] Verificar coherencia: ¿el código 7 en transparencia es el mismo banco que `00007` en el scraper?

---

## 8. Notebook 02 — EDA

### 8.1 Visualizaciones mínimas requeridas

| # | Gráfico | Variables |
|---|---------|-----------|
| 1 | Concentración del sistema (Pareto) | Activo total por banco |
| 2 | Box plot rentabilidad por tipo | ROE proxy, ROA proxy × tipo_entidad |
| 3 | Box plot riesgo crediticio por tipo | % cartera irregular × tipo_entidad |
| 4 | Distribución del tamaño | Log(Activo total) |
| 5 | Mapa de missings | dataset_master completo |
| 6 | Heatmap de correlación | Variables candidatas al clustering |
| 7 | Heatmap de oferta de productos | bancos × productos (binario: ofrece/no ofrece) |
| 8 | TEA comparativa por tipo de banco | tea_max por producto × tipo_entidad |

### 8.2 Variables candidatas para el clustering

**Del scraper — ratios a calcular (adimensionales, independientes de inflación):**
- `prestamos_sprivado / activo` — intermediación crediticia
- `titulos_publicos / activo` — exposición soberana
- `depositos_sprivado / pasivo` — fondeo minorista
- `patrimonio_neto / activo` — solvencia / apalancamiento inverso
- `resultado / activo` — ROA proxy
- `log(activo)` — tamaño

**Del scraper — riesgo (ya en %):**
- Suma de cartera en situación 3+4+5 — irregularidad total
- `previsiones / financiaciones` — cobertura de incobrables

**Del scraper — estructura operativa:**
- `dotacion_personal` — escala
- Cantidad total de sucursales

**Del Régimen de Transparencia — oferta de productos (binarias):**
- `ofrece_hipoteca`, `hipoteca_uva`
- `ofrece_personal`, `ofrece_prendario`
- `ofrece_pfijo`, `pfijo_uva`
- `ofrece_paquete`, `paquete_tiene_premium`
- `ofrece_tarjeta`, `tarjeta_tiene_premium`
- `caja_apertura_simplificada`

**Del Régimen de Transparencia — precio de productos (numérico):**
- `hipoteca_tea_max` — costo del crédito hipotecario
- `personal_tea_max` — costo del crédito personal
- `tarjeta_tea_financiacion` — costo del crédito revolving
- `pfijo_tea_min` — rendimiento ofrecido al ahorrista

---

## 9. Particularidades técnicas

**Scraper (BCRA portal):**
- Output en formato **long** → siempre pivotear antes de analizar
- Valores de estados contables en **miles de pesos corrientes** — usar ratios para el clustering
- `indicadores_economicos` puede estar vacío (JS) — si es así, calcular ROE/ROA desde estados contables

**Régimen de Transparencia:**
- Encoding obligatorio: `latin-1` — si se lee como UTF-8 rompe caracteres españoles
- Decimales con `,` (coma) → convertir a `.` antes de cualquier operación numérica
- `codigo_entidad` viene como integer en el CSV → normalizar a string de 5 dígitos con `zfill(5)`
- Un banco ausente de un archivo = **no ofrece ese producto** → codificar como 0, no como NaN
- Los datos de transparencia son snapshots con fechas variables por banco (columna `Fecha de Información`) — no es una serie de tiempo, es el estado actual al momento de la carga
- TARJETAS tiene 139 entidades porque incluye emisores no bancarios (tarjetas de retail, etc.) — filtrar a los 56 bancos antes de agregar

**Join entre fuentes:**
- Clave: `codigo_entidad` en formato string de 5 dígitos con ceros (`"00007"`, `"00011"`, etc.)
- El scraper usa `codigo_bcra` con el mismo formato → renombrar al hacer el merge

---

## 10. Flujo de trabajo para Claude Code

```
PASO 1 — Setup
  └─ Crear estructura de carpetas (Sección 5)
  └─ pip install -r requirements.txt

PASO 2 — Scraping
  └─ Copiar scripts/00_scraper_bcra.py al proyecto
  └─ python scripts/00_scraper_bcra.py
  └─ Verificar data/processed/bcra_bancos.db

PASO 3 — Colocar archivos de transparencia
  └─ Copiar CSVs a data/raw/transparencia/
  └─ Archivos: HIPOTECA.CSV, PERSONALES.CSV, PRENDARIOS.CSV,
               PFIJO.CSV, PAQUETE.CSV, TARJETAS.CSV, CAJADEAHORROS.CSV

PASO 4 — Notebook 01: ingesta
  └─ Cargar SQLite → DataFrames → pivotear
  └─ Cargar CSVs de transparencia → normalizar → agregar por banco
  └─ Join → dataset_master.csv

PASO 5 — Notebook 02: EDA
  └─ Calidad de datos (missings, tipos, outliers)
  └─ Estadísticas descriptivas y visualizaciones (Sección 8.1)
  └─ Conclusiones: variables candidatas y decisiones para Entrega III
```

**Prompt para iniciar Claude Code:**
> "Leé el SPEC.md. El scraper `scripts/00_scraper_bcra.py` ya existe y los CSVs de transparencia están en `data/raw/transparencia/`. Empecemos con el notebook 01 de ingesta."

---

*Actualizar este SPEC al confirmar los períodos reales del scraper y la cobertura final de bancos.*
