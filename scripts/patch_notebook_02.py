"""Patch del notebook 02_eda.ipynb — aplica bugfixes y agrega sección 8.

Uso:
    python scripts/patch_notebook_02.py

Modifica en sitio `notebooks/02_eda.ipynb`:
- Celda de construir_ratios: fija signo de eficiencia y cartera irregular (NaN si no aplica).
- Inserta sección 8 (validaciones + Kruskal-Wallis + HHI + missings) antes de la sección 6 de artefactos.
- Actualiza la celda de artefactos para exportar también oferta_banco y variedad.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "02_eda.ipynb"


def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in src.splitlines()][:-1] + [src.splitlines()[-1]]
        if src.strip()
        else [],
    }


def md_cell(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in src.splitlines()][:-1] + [src.splitlines()[-1]]
        if src.strip()
        else [],
    }


NEW_CONSTRUIR_RATIOS = '''def construir_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Crea ratios bancarios estándar a partir de las columnas de balance.

    Los nombres originales vienen con espacios (ej. 'A C T I V O') porque el BCRA
    los publica así en las tablas HTML. Acá los renombramos a algo manejable.
    """
    df = df.copy()

    rename = {
        "A C T I V O": "activo",
        "P A S I V O": "pasivo",
        "P A T R I M O N I O N E T O": "patrimonio",
        "R D O S. I N T E G R A L E S A C U M. D E L P E R I O D O": "resultado",
        "DEPÓSITOS": "depositos",
        "PRÉSTAMOS": "prestamos",
        "EFECTIVO Y DEPOSITO EN BANCOS": "efectivo",
        "INGRESOS FINANCIEROS": "ing_fin",
        "EGRESOS FINANCIEROS": "egr_fin",
        "INGRESOS POR SERVICIOS": "ing_serv",
        "EGRESOS POR SERVICIOS": "egr_serv",
        "GASTOS DE ADMINISTRACIÓN": "gastos_adm",
        "Dotación de personal": "dotacion",
        "TOTAL DE FINANCIACIONES Y GARANTIAS OTORGADAS ($)": "financiaciones_totales",
    }
    df = df.rename(columns=rename)

    # Ratios de rentabilidad
    df["roa"] = df["resultado"] / df["activo"] * 100
    df["roe"] = df["resultado"] / df["patrimonio"] * 100

    # Estructura financiera
    df["apalancamiento"] = df["activo"] / df["patrimonio"]
    df["depositos_sobre_activo"] = df["depositos"] / df["activo"] * 100
    df["prestamos_sobre_activo"] = df["prestamos"] / df["activo"] * 100
    df["liquidez"] = df["efectivo"] / df["depositos"] * 100

    # Eficiencia (gastos adm / (margen financiero + ing servicios netos)).
    # Las columnas de egresos y gastos vienen con signo NEGATIVO del scraping
    # (así las publica el BCRA en los estados contables). Por eso usamos abs()
    # para trabajar con magnitudes y que el ratio quede en el rango estándar
    # ~40-80% positivo.
    margen = (df["ing_fin"] - df["egr_fin"].abs()) + (df["ing_serv"] - df["egr_serv"].abs())
    df["eficiencia"] = df["gastos_adm"].abs() / margen * 100

    # Cartera irregular = situaciones 3 + 4 + 5 sobre total de financiaciones.
    # Importante: si el banco no tiene financiaciones reportadas (mayoristas,
    # de inversión, sucursales chicas) dejamos NaN en vez de 0, porque "0% de
    # cartera mala" es distinto a "no aplica / no tiene cartera minorista".
    sit_cols = [
        "TF.Sit.3: Con problemas/Riesgo medio (%)",
        "TF.Sit.4: Con alto riesgo de insolvencia/Riesgo alto (%)",
        "TF.Sit.5: Irrecuperable (%)",
    ]
    for c in sit_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    cartera = df[sit_cols].sum(axis=1, min_count=1)
    if "financiaciones_totales" in df.columns:
        fin_tot = pd.to_numeric(df["financiaciones_totales"], errors="coerce")
        cartera = cartera.where(fin_tot > 0)
    df["cartera_irregular"] = cartera

    # Activo por empleado — proxy de productividad
    df["activo_por_empleado"] = df["activo"] / df["dotacion"]

    return df


masters = {}
for corte in CORTES:
    raw = pd.read_csv(PROC / f"dataset_master_{corte}.csv")
    raw["corte"] = CORTE_LABEL[corte]
    masters[corte] = construir_ratios(raw)

# Panel transversal (long) para análisis de evolución
panel = pd.concat(masters.values(), ignore_index=True)
print("Panel transversal:", panel.shape)
panel[["corte", "tipo_entidad"]].value_counts().unstack(fill_value=0)'''


NEW_ARTEFACTOS = '''out_panel = PROC / "panel_ratios.csv"
panel.to_csv(out_panel, index=False)
print(f"Guardado: {out_panel}  shape={panel.shape}")

# Exportamos también la oferta agregada y la variedad — el notebook 03 los
# necesita para sumar features de transparencia al clustering.
out_oferta = PROC / "oferta_banco.csv"
oferta_banco.to_csv(out_oferta)
print(f"Guardado: {out_oferta}  shape={oferta_banco.shape}")

out_variedad = PROC / "variedad_productos.csv"
variedad.to_csv(out_variedad)
print(f"Guardado: {out_variedad}  shape={variedad.shape}")'''


# Nuevas celdas de la sección 8
SECTION_8_CELLS: list[dict] = [
    md_cell(
        "## 8. Validaciones y análisis complementarios\n\n"
        "Antes de pasar al clustering (notebook 03) cerramos tres frentes:\n\n"
        "- **Bugs detectados en la primera pasada del EDA** (signo de la eficiencia, "
        "anualización del resultado en Dic-2025, cartera irregular de extranjeros).\n"
        "- **Respaldo estadístico** para la afirmación \"los tres tipos de entidad "
        "son distinguibles\" — Kruskal-Wallis por ratio.\n"
        "- **Concentración del sistema** (HHI, share del top-5) y diagnóstico de "
        "missings por banco, para decidir exclusiones del clustering."
    ),
    md_cell(
        "### 8.1 Inspección de signos crudos en el estado contable\n\n"
        "En la primera corrida la eficiencia daba negativa para los tres tipos "
        "(-13% / -17% / -20%) cuando el ratio estándar vive en ~40-80% positivo. "
        "Inspeccionamos los valores crudos de las cuatro columnas del margen para "
        "tres bancos representativos — confirmamos que el BCRA publica los egresos "
        "y gastos con signo negativo, así que el fix está en tomar valor absoluto "
        "antes de combinar."
    ),
    code_cell(
        '''cols_signo = [
    "nombre_banco", "tipo_entidad",
    "INGRESOS FINANCIEROS", "EGRESOS FINANCIEROS",
    "INGRESOS POR SERVICIOS", "EGRESOS POR SERVICIOS",
    "GASTOS DE ADMINISTRACIÓN",
    "R D O S. I N T E G R A L E S A C U M. D E L P E R I O D O",
]
raw24 = pd.read_csv(PROC / "dataset_master_dic2024.csv")
muestra = raw24[raw24["nombre_banco"].str.contains(
    "NACION|SANTANDER|BNP", case=False, na=False)][cols_signo]
muestra.set_index("nombre_banco").T'''
    ),
    md_cell(
        "Los egresos financieros, los egresos por servicios y los gastos de "
        "administración vienen con signo negativo (entre paréntesis en las tablas "
        "HTML del BCRA). En `construir_ratios` aplicamos `abs()` sobre esas "
        "columnas antes de combinarlas, lo que devuelve la eficiencia al rango "
        "esperado (ver tabla de medianas por tipo en §4.2)."
    ),
    md_cell(
        "### 8.2 ¿El resultado de Dic-2025 está anualizado?\n\n"
        "La caída de ROA entre Dic-24 y Dic-25 parecía sospechosa. Si el campo "
        "`R D O S. I N T E G R A L E S A C U M. D E L P E R I O D O` viniera "
        "mensualizado en Dic-25, deberíamos observar un ratio sistemático de "
        "~11/12 ≈ 0.92 en el resultado Dic-25 / Dic-24 para los bancos más "
        "grandes. Chequeamos el top-5 por activo."
    ),
    code_cell(
        '''top5 = masters["dic2024"].nlargest(5, "activo")[
    ["codigo_entidad", "nombre_banco", "activo", "resultado", "ing_fin"]
].copy()
top5 = top5.rename(columns={"activo": "act_24", "resultado": "res_24", "ing_fin": "ing_24"})

d25_flat = masters["dic2025"][["codigo_entidad", "activo", "resultado", "ing_fin"]].rename(
    columns={"activo": "act_25", "resultado": "res_25", "ing_fin": "ing_25"}
)
chk_2025 = top5.merge(d25_flat, on="codigo_entidad")
chk_2025["ratio_res_25_24"] = chk_2025["res_25"] / chk_2025["res_24"]
chk_2025["ratio_ing_25_24"] = chk_2025["ing_25"] / chk_2025["ing_24"]
chk_2025["ratio_act_25_24"] = chk_2025["act_25"] / chk_2025["act_24"]
chk_2025[["nombre_banco", "res_24", "res_25",
          "ratio_res_25_24", "ratio_ing_25_24", "ratio_act_25_24"]].round(3)'''
    ),
    md_cell(
        "Los ratios `res_25/res_24` son muy heterogéneos entre bancos "
        "(del orden de 0.07 a 6.5) y no se concentran en ~0.92. Descartamos el "
        "escenario \"Dic-2025 viene en 11/12 avos\": el corte que trae el BCRA "
        "en `dataset_master_dic2025.csv` corresponde al cierre anual, y la "
        "caída del ROA mediana entre 2024 y 2025 es un resultado real del "
        "sistema, no un artefacto de anualización."
    ),
    md_cell(
        "### 8.3 Cartera irregular en extranjeros — diagnóstico\n\n"
        "La mediana de cartera irregular de los 9 extranjeros daba exactamente "
        "0%. Antes de leer eso como \"mejor calidad de cartera\", verificamos "
        "cuántos de esos bancos efectivamente tienen cartera minorista. Los que "
        "no tienen financiaciones reportadas (bancos mayoristas / de inversión) "
        "no deben entrar al denominador del grupo."
    ),
    code_cell(
        '''ext24 = masters["dic2024"][masters["dic2024"]["tipo_entidad"] == "extranjero"].copy()
diag_ext = ext24[[
    "nombre_banco",
    "financiaciones_totales",
    "TF.Sit.3: Con problemas/Riesgo medio (%)",
    "TF.Sit.4: Con alto riesgo de insolvencia/Riesgo alto (%)",
    "TF.Sit.5: Irrecuperable (%)",
    "cartera_irregular",
]].copy()
diag_ext["aplica"] = diag_ext["financiaciones_totales"].fillna(0) > 0
diag_ext.sort_values("aplica", ascending=False)'''
    ),
    md_cell(
        "De los 9 extranjeros, solo los que tienen `financiaciones_totales > 0` "
        "aportan una lectura válida de cartera irregular. A los restantes "
        "(mayoristas/inversión) `construir_ratios` les asigna `NaN` en "
        "`cartera_irregular`, de modo que la mediana del grupo se calcula solo "
        "sobre los bancos donde la métrica aplica. Recalculamos acá la mediana "
        "por tipo con la corrección ya aplicada:"
    ),
    code_cell(
        '''masters["dic2024"].groupby("tipo_entidad")["cartera_irregular"].agg(
    ["count", "median", "mean"]
).reindex(ORDEN_TIPO).round(2)'''
    ),
    md_cell(
        "### 8.4 Kruskal-Wallis — ¿los tres tipos de entidad son distinguibles?\n\n"
        "Con n=9 extranjeros conviene un test no paramétrico. Corremos "
        "Kruskal-Wallis por variable sobre Dic-2024 — la hipótesis nula es "
        "\"las tres distribuciones vienen del mismo población\". p-valor "
        "bajo ⇒ al menos un tipo difiere."
    ),
    code_cell(
        '''from scipy import stats

VARS_KW = [
    "roa", "roe", "apalancamiento",
    "depositos_sobre_activo", "cartera_irregular",
    "liquidez", "activo_por_empleado", "n_productos_ofrecidos",
]

filas = []
for var in VARS_KW:
    grupos = [
        df24.loc[df24["tipo_entidad"] == t, var].dropna().values
        for t in ORDEN_TIPO
    ]
    n_por_grupo = [len(g) for g in grupos]
    if all(len(g) >= 2 for g in grupos):
        H, p = stats.kruskal(*grupos)
    else:
        H, p = float("nan"), float("nan")
    filas.append({
        "variable": var,
        "n_publico": n_por_grupo[0],
        "n_privado_nac": n_por_grupo[1],
        "n_extranjero": n_por_grupo[2],
        "H": round(H, 2) if pd.notna(H) else None,
        "p_valor": round(p, 4) if pd.notna(p) else None,
        "significativo_5pct": (p < 0.05) if pd.notna(p) else None,
    })
kw = pd.DataFrame(filas)
kw'''
    ),
    md_cell(
        "Las variables con p < 0.05 son las que efectivamente separan perfiles "
        "y son buenas candidatas a entrar al clustering con peso pleno. Las que "
        "dan p > 0.05 no necesariamente se descartan (pueden aportar en "
        "combinación con otras), pero no respaldan por sí solas la afirmación "
        "de que los tres perfiles difieran en esa dimensión."
    ),
    md_cell(
        "### 8.5 Concentración del sistema — HHI y share del top-5\n\n"
        "Dos medidas clásicas de concentración sobre el activo:\n\n"
        "- **HHI**: suma de cuadrados de los shares individuales. Convención "
        "DoJ: <1500 mercado competitivo, 1500-2500 moderadamente concentrado, "
        ">2500 altamente concentrado. Lo expresamos en base 10000.\n"
        "- **Share del top-5**: qué porción del activo del sistema explican "
        "los 5 bancos más grandes.\n\n"
        "Mostramos los tres cortes para ver si la concentración se movió."
    ),
    code_cell(
        '''def hhi_y_top5(df: pd.DataFrame) -> dict:
    tot = df["activo"].sum()
    shares = df["activo"] / tot
    hhi = float((shares ** 2).sum() * 10000)
    top5_share = float(df.nlargest(5, "activo")["activo"].sum() / tot * 100)
    return {"HHI": round(hhi, 1), "top5_share_pct": round(top5_share, 2)}


concentracion = pd.DataFrame(
    {CORTE_LABEL[c]: hhi_y_top5(masters[c]) for c in CORTES}
).T
concentracion'''
    ),
    md_cell(
        "### 8.6 Missings por banco — ¿quién conviene excluir del clustering?\n\n"
        "El análisis por variable de la §1 muestra qué columnas tienen más "
        "huecos. Acá invertimos la mirada: sobre cada banco, qué fracción del "
        "set candidato al clustering (ratios + oferta comercial) está vacía. "
        "Los bancos con demasiados missings van a traer ruido a cualquier "
        "imputación."
    ),
    code_cell(
        '''VARS_CLUSTER = [
    "roa", "roe", "apalancamiento",
    "depositos_sobre_activo", "prestamos_sobre_activo",
    "liquidez", "eficiencia", "cartera_irregular",
    "activo_por_empleado", "n_productos_ofrecidos",
]

# Sumamos features de oferta agregada (se calculan más abajo en §7.6, así que
# los traemos desde oferta_banco si ya está disponible).
cruce24 = df24.set_index("codigo_entidad").copy()
if "oferta_banco" in globals():
    cruce24 = cruce24.join(oferta_banco, how="left")
    VARS_CLUSTER_FULL = VARS_CLUSTER + list(oferta_banco.columns)
else:
    VARS_CLUSTER_FULL = VARS_CLUSTER

miss_banco = cruce24[VARS_CLUSTER_FULL].isna().mean(axis=1) * 100
miss_banco = miss_banco.to_frame("pct_missing").join(
    cruce24[["nombre_banco", "tipo_entidad"]]
)
top10_missings = miss_banco.sort_values("pct_missing", ascending=False).head(10)
top10_missings.round(1)'''
    ),
    md_cell(
        "**Criterio de exclusión propuesto para el notebook 03**: bancos con "
        "más del 40% de missings sobre el set del clustering. Son típicamente "
        "sucursales mayoristas / de inversión (BNP, JPMorgan, Bank of China, "
        "Citelem) que no reportan cartera minorista ni publican productos en "
        "transparencia — no tiene sentido forzarlos a un perfil junto a "
        "bancos retail. La decisión final queda documentada en el notebook 03 "
        "y se ejecuta al armar la matriz de features."
    ),
    code_cell(
        '''umbral_exclusion = 40
excluir = miss_banco[miss_banco["pct_missing"] > umbral_exclusion]
print(f"Bancos con >{umbral_exclusion}% de missings sobre {len(VARS_CLUSTER_FULL)} "
      f"variables candidatas:")
excluir.sort_values("pct_missing", ascending=False).round(1)'''
    ),
]


def main() -> None:
    nb = json.loads(NB_PATH.read_text())

    # --- Reemplazo celda 3 (construir_ratios) ---
    cell3 = nb["cells"][3]
    cell3["source"] = [ln + "\n" for ln in NEW_CONSTRUIR_RATIOS.split("\n")]
    # Sin trailing \n en la última línea
    cell3["source"][-1] = cell3["source"][-1].rstrip("\n")
    cell3["outputs"] = []
    cell3["execution_count"] = None

    # --- Reemplazo celda de artefactos (última code cell, índice 35) ---
    # Localizamos por contenido, por si el índice cambia
    art_idx = None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code" and "panel_ratios.csv" in "".join(c["source"]):
            art_idx = i
            break
    if art_idx is None:
        raise SystemExit("No encontré la celda de artefactos")
    nb["cells"][art_idx]["source"] = [ln + "\n" for ln in NEW_ARTEFACTOS.split("\n")]
    nb["cells"][art_idx]["source"][-1] = nb["cells"][art_idx]["source"][-1].rstrip("\n")
    nb["cells"][art_idx]["outputs"] = []
    nb["cells"][art_idx]["execution_count"] = None

    # Encontramos el header "## 6. Artefactos" (markdown, justo antes de art_idx)
    # e insertamos la sección 8 ANTES de ese header.
    header6_idx = None
    for i in range(art_idx - 1, -1, -1):
        c = nb["cells"][i]
        if c["cell_type"] == "markdown" and "6. Artefactos" in "".join(c["source"]):
            header6_idx = i
            break
    if header6_idx is None:
        raise SystemExit("No encontré el header '## 6. Artefactos'")

    # Ensamblamos las nuevas celdas con source terminando sin \n en la última línea
    new_cells = []
    for c in SECTION_8_CELLS:
        cc = copy.deepcopy(c)
        if cc["source"]:
            # source ya viene como lista de líneas, cada una con \n, salvo la última
            # por cómo la construimos arriba. Normalizamos:
            txt = "".join(cc["source"])
            cc["source"] = [ln + "\n" for ln in txt.split("\n")]
            cc["source"][-1] = cc["source"][-1].rstrip("\n")
        new_cells.append(cc)

    nb["cells"] = nb["cells"][:header6_idx] + new_cells + nb["cells"][header6_idx:]

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"Notebook actualizado: {NB_PATH}")
    print(f"Total de celdas: {len(nb['cells'])}")


if __name__ == "__main__":
    main()
