"""Inserta la sección 7 (análisis de transparencia) en notebook 02, antes de la sección 6."""
import json
from pathlib import Path

NB_PATH = Path("notebooks/02_eda.ipynb")
nb = json.loads(NB_PATH.read_text())


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


nuevas = [
    md(
        "## 7. Oferta comercial — Régimen de Transparencia\n"
        "\n"
        "Los CSVs de transparencia (`data/raw/transparencia/`) traen los productos que cada banco **ofrece públicamente**: "
        "hipotecas, personales, prendarios, plazo fijo, tarjetas y paquetes. En el notebook 01 los agregamos a nivel banco "
        "para unirlos al master (tasa max, monto max, comisión min, etc.), pero aquí explotamos:\n"
        "\n"
        "- **Distribuciones producto-nivel** (no banco-nivel): la dispersión de tasas dentro de un banco es parte del perfil.\n"
        "- **Requisitos de acceso**: ingreso mínimo y antigüedad laboral — proxy de a qué público apunta cada banco.\n"
        "- **Segmentación**: tarjetas y paquetes premium vs clásicas.\n"
        "- **Cruce oferta vs balance**: ¿los bancos más rentables cobran tasas más altas?"
    ),
    md("### 7.1 Carga de CSVs producto-nivel"),
    code(
        "RAW_TR = Path(\"../data/raw/transparencia\")\n"
        "\n"
        "# Leemos cada CSV y le adjuntamos el tipo_entidad desde el master.\n"
        "# Filtramos por las 56 entidades scrapeadas del BCRA (las mismas del master).\n"
        "tipos_por_codigo = df24.set_index(\"codigo_entidad\")[\"tipo_entidad\"].to_dict()\n"
        "nombres_por_codigo = df24.set_index(\"codigo_entidad\")[\"nombre_banco\"].to_dict()\n"
        "\n"
        "\n"
        "def cargar_tr(nombre_archivo: str, col_tasa: str | None = None,\n"
        "              col_comision: str | None = None) -> pd.DataFrame:\n"
        "    df = pd.read_csv(RAW_TR / nombre_archivo, sep=\";\", encoding=\"latin-1\")\n"
        "    df = df.rename(columns={\"Código de Entidad\": \"codigo_entidad\"})\n"
        "    df = df[df[\"codigo_entidad\"].isin(tipos_por_codigo)].copy()\n"
        "    df[\"tipo_entidad\"] = df[\"codigo_entidad\"].map(tipos_por_codigo)\n"
        "    df[\"nombre_banco\"] = df[\"codigo_entidad\"].map(nombres_por_codigo)\n"
        "    if col_tasa and col_tasa in df.columns:\n"
        "        df[col_tasa] = pd.to_numeric(df[col_tasa].astype(str).str.replace(\",\", \".\"), errors=\"coerce\")\n"
        "    if col_comision and col_comision in df.columns:\n"
        "        df[col_comision] = pd.to_numeric(df[col_comision].astype(str).str.replace(\",\", \".\"), errors=\"coerce\")\n"
        "    return df\n"
        "\n"
        "\n"
        "tr_hipoteca = cargar_tr(\"HIPOTECA.CSV\", \"Tasa efectiva anual máxima\")\n"
        "tr_personal = cargar_tr(\"PERSONALES.CSV\", \"Tasa efectiva anual máxima\")\n"
        "tr_prendario = cargar_tr(\"PRENDARIOS.CSV\", \"Tasa efectiva anual máxima\")\n"
        "tr_pfijo = cargar_tr(\"PFIJO.CSV\", \"Tasa efectiva anual mínima\")\n"
        "tr_tarjetas = cargar_tr(\"TARJETAS.CSV\",\n"
        "                         \"Tasa efectiva anual máxima de interés compensatorio por financiación de saldos\",\n"
        "                         \"Comisión máxima por administración y mantenimiento de la cuenta\")\n"
        "tr_paquete = cargar_tr(\"PAQUETE.CSV\", None,\n"
        "                        \"Comisión máxima por servicio de mantenimiento de paquete\")\n"
        "\n"
        "print(f\"Hipoteca: {len(tr_hipoteca)} productos de {tr_hipoteca['codigo_entidad'].nunique()} bancos\")\n"
        "print(f\"Personal: {len(tr_personal)} productos de {tr_personal['codigo_entidad'].nunique()} bancos\")\n"
        "print(f\"Prendario: {len(tr_prendario)} productos de {tr_prendario['codigo_entidad'].nunique()} bancos\")\n"
        "print(f\"Plazo fijo: {len(tr_pfijo)} productos de {tr_pfijo['codigo_entidad'].nunique()} bancos\")\n"
        "print(f\"Tarjetas: {len(tr_tarjetas)} productos de {tr_tarjetas['codigo_entidad'].nunique()} bancos\")\n"
        "print(f\"Paquete: {len(tr_paquete)} productos de {tr_paquete['codigo_entidad'].nunique()} bancos\")"
    ),
    md(
        "### 7.2 Distribución de tasas por producto y tipo de entidad\n"
        "\n"
        "Cada punto es **un producto** (no un banco). Así se ve la dispersión real que enfrenta un cliente cuando shopea."
    ),
    code(
        "TASAS = [\n"
        "    (tr_hipoteca, \"Tasa efectiva anual máxima\", \"Hipoteca — TEA máx\"),\n"
        "    (tr_personal, \"Tasa efectiva anual máxima\", \"Personal — TEA máx\"),\n"
        "    (tr_prendario, \"Tasa efectiva anual máxima\", \"Prendario — TEA máx\"),\n"
        "    (tr_pfijo, \"Tasa efectiva anual mínima\", \"Plazo fijo — TEA mín\"),\n"
        "    (tr_tarjetas,\n"
        "     \"Tasa efectiva anual máxima de interés compensatorio por financiación de saldos\",\n"
        "     \"Tarjeta — TEA financiación\"),\n"
        "]\n"
        "\n"
        "fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n"
        "for ax, (df_tr, col, titulo) in zip(axes.flat, TASAS):\n"
        "    sub = df_tr.dropna(subset=[col])\n"
        "    sns.boxplot(data=sub, x=\"tipo_entidad\", y=col, order=ORDEN_TIPO,\n"
        "                palette=PALETA, ax=ax)\n"
        "    sns.stripplot(data=sub, x=\"tipo_entidad\", y=col, order=ORDEN_TIPO,\n"
        "                  color=\"black\", size=2.5, alpha=0.3, ax=ax)\n"
        "    ax.set_title(titulo)\n"
        "    ax.set_xlabel(\"\")\n"
        "    ax.set_ylabel(\"TEA (%)\")\n"
        "axes[1, 2].axis(\"off\")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "### 7.3 Requisitos de acceso\n"
        "\n"
        "¿Cuánto ingreso mínimo y qué antigüedad laboral exigen los bancos para sus productos? "
        "Agregamos por banco (mediana de sus productos) y comparamos por tipo."
    ),
    code(
        "# Para préstamos personales y tarjetas — productos de retail más comparables.\n"
        "def exigencia_por_banco(df_tr: pd.DataFrame) -> pd.DataFrame:\n"
        "    cols = [\"Ingreso mínimo mensual solicitado\", \"Antigüedad laboral mínima (meses)\"]\n"
        "    for c in cols:\n"
        "        if c in df_tr.columns:\n"
        "            df_tr[c] = pd.to_numeric(df_tr[c].astype(str).str.replace(\",\", \".\"), errors=\"coerce\")\n"
        "    return df_tr.groupby([\"codigo_entidad\", \"tipo_entidad\"])[cols].median().reset_index()\n"
        "\n"
        "\n"
        "exig_personal = exigencia_por_banco(tr_personal.copy())\n"
        "exig_tarjetas = exigencia_por_banco(tr_tarjetas.copy())\n"
        "\n"
        "fig, axes = plt.subplots(2, 2, figsize=(14, 9))\n"
        "for ax, (df_ex, titulo) in zip(\n"
        "    axes[0], [(exig_personal, \"Personal\"), (exig_tarjetas, \"Tarjetas\")]\n"
        "):\n"
        "    sns.boxplot(data=df_ex, x=\"tipo_entidad\", y=\"Ingreso mínimo mensual solicitado\",\n"
        "                order=ORDEN_TIPO, palette=PALETA, ax=ax)\n"
        "    ax.set_title(f\"{titulo} — ingreso mínimo exigido (mediana por banco)\")\n"
        "    ax.set_xlabel(\"\")\n"
        "for ax, (df_ex, titulo) in zip(\n"
        "    axes[1], [(exig_personal, \"Personal\"), (exig_tarjetas, \"Tarjetas\")]\n"
        "):\n"
        "    sns.boxplot(data=df_ex, x=\"tipo_entidad\", y=\"Antigüedad laboral mínima (meses)\",\n"
        "                order=ORDEN_TIPO, palette=PALETA, ax=ax)\n"
        "    ax.set_title(f\"{titulo} — antigüedad laboral exigida (meses)\")\n"
        "    ax.set_xlabel(\"\")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "### 7.4 Segmentación — tarjetas y paquetes premium\n"
        "\n"
        "El campo `Segmento` clasifica la oferta (clásica / gold / platinum / black / joven / etc.). "
        "Miramos la proporción de oferta premium por tipo de entidad."
    ),
    code(
        "def clasificar_premium(seg: str) -> str:\n"
        "    if not isinstance(seg, str):\n"
        "        return \"sin_dato\"\n"
        "    s = seg.lower()\n"
        "    if any(k in s for k in [\"platinum\", \"black\", \"signature\", \"infinite\", \"premium\"]):\n"
        "        return \"premium\"\n"
        "    if \"gold\" in s or \"oro\" in s:\n"
        "        return \"gold\"\n"
        "    return \"clasica\"\n"
        "\n"
        "\n"
        "tr_tarjetas[\"categoria\"] = tr_tarjetas[\"Segmento\"].apply(clasificar_premium)\n"
        "tr_paquete[\"categoria\"] = tr_paquete[\"Segmento\"].apply(clasificar_premium)\n"
        "\n"
        "mix_tarjetas = (\n"
        "    tr_tarjetas.groupby([\"tipo_entidad\", \"categoria\"]).size()\n"
        "    .unstack(fill_value=0)\n"
        "    .reindex(ORDEN_TIPO)\n"
        ")\n"
        "mix_paquete = (\n"
        "    tr_paquete.groupby([\"tipo_entidad\", \"categoria\"]).size()\n"
        "    .unstack(fill_value=0)\n"
        "    .reindex(ORDEN_TIPO)\n"
        ")\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "(mix_tarjetas.div(mix_tarjetas.sum(axis=1), axis=0) * 100).plot(\n"
        "    kind=\"bar\", stacked=True, ax=axes[0], colormap=\"viridis\"\n"
        ")\n"
        "axes[0].set_title(\"Tarjetas — mix por categoría (%)\")\n"
        "axes[0].set_ylabel(\"% de productos\")\n"
        "axes[0].set_xlabel(\"\")\n"
        "axes[0].legend(title=\"\", bbox_to_anchor=(1.01, 1), loc=\"upper left\")\n"
        "axes[0].tick_params(axis=\"x\", rotation=0)\n"
        "(mix_paquete.div(mix_paquete.sum(axis=1), axis=0) * 100).plot(\n"
        "    kind=\"bar\", stacked=True, ax=axes[1], colormap=\"viridis\"\n"
        ")\n"
        "axes[1].set_title(\"Paquetes — mix por categoría (%)\")\n"
        "axes[1].set_ylabel(\"% de productos\")\n"
        "axes[1].set_xlabel(\"\")\n"
        "axes[1].legend(title=\"\", bbox_to_anchor=(1.01, 1), loc=\"upper left\")\n"
        "axes[1].tick_params(axis=\"x\", rotation=0)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "print(\"Cantidades absolutas — tarjetas:\")\n"
        "print(mix_tarjetas)\n"
        "print(\"\\nCantidades absolutas — paquetes:\")\n"
        "print(mix_paquete)"
    ),
    md(
        "### 7.5 Variedad de oferta — productos únicos por banco\n"
        "\n"
        "No es lo mismo \"ofrecer préstamos personales\" que ofrecer 15 variantes distintas. "
        "Contamos productos únicos por banco en cada categoría."
    ),
    code(
        "variedad = pd.DataFrame({\n"
        "    \"hipoteca\": tr_hipoteca.groupby(\"codigo_entidad\").size(),\n"
        "    \"personal\": tr_personal.groupby(\"codigo_entidad\").size(),\n"
        "    \"prendario\": tr_prendario.groupby(\"codigo_entidad\").size(),\n"
        "    \"pfijo\": tr_pfijo.groupby(\"codigo_entidad\").size(),\n"
        "    \"tarjetas\": tr_tarjetas.groupby(\"codigo_entidad\").size(),\n"
        "    \"paquetes\": tr_paquete.groupby(\"codigo_entidad\").size(),\n"
        "}).fillna(0).astype(int)\n"
        "variedad[\"tipo_entidad\"] = variedad.index.map(tipos_por_codigo)\n"
        "\n"
        "# Medianas por tipo\n"
        "variedad_mediana = variedad.groupby(\"tipo_entidad\").median().reindex(ORDEN_TIPO)\n"
        "variedad_mediana"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(10, 5))\n"
        "variedad_mediana.plot(kind=\"bar\", ax=ax, colormap=\"tab10\")\n"
        "ax.set_title(\"Variedad de oferta — mediana de productos por banco\")\n"
        "ax.set_ylabel(\"Productos ofrecidos (mediana)\")\n"
        "ax.set_xlabel(\"\")\n"
        "ax.legend(title=\"Categoría\", bbox_to_anchor=(1.01, 1), loc=\"upper left\")\n"
        "ax.tick_params(axis=\"x\", rotation=0)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "### 7.6 Cruce: oferta comercial vs balance\n"
        "\n"
        "¿Los bancos más rentables o más grandes exhiben tasas más altas, cobran más comisiones o exigen más? "
        "Unimos las tasas y comisiones agregadas a nivel banco con los ratios del master."
    ),
    code(
        "oferta_banco = pd.DataFrame({\n"
        "    \"tea_personal_mediana\": tr_personal.groupby(\"codigo_entidad\")[\"Tasa efectiva anual máxima\"].median(),\n"
        "    \"tea_tarjeta_mediana\": tr_tarjetas.groupby(\"codigo_entidad\")[\n"
        "        \"Tasa efectiva anual máxima de interés compensatorio por financiación de saldos\"\n"
        "    ].median(),\n"
        "    \"comision_tarjeta_mediana\": tr_tarjetas.groupby(\"codigo_entidad\")[\n"
        "        \"Comisión máxima por administración y mantenimiento de la cuenta\"\n"
        "    ].median(),\n"
        "    \"ingreso_min_personal\": tr_personal.groupby(\"codigo_entidad\")[\n"
        "        \"Ingreso mínimo mensual solicitado\"\n"
        "    ].median(),\n"
        "    \"variedad_total\": variedad[[\"hipoteca\", \"personal\", \"prendario\",\n"
        "                                  \"pfijo\", \"tarjetas\", \"paquetes\"]].sum(axis=1),\n"
        "})\n"
        "\n"
        "cruce = df24[[\"codigo_entidad\", \"nombre_banco\", \"tipo_entidad\",\n"
        "              \"activo\", \"roa\", \"roe\", \"eficiencia\",\n"
        "              \"cartera_irregular\"]].merge(\n"
        "    oferta_banco, left_on=\"codigo_entidad\", right_index=True, how=\"left\"\n"
        ")\n"
        "\n"
        "cols_corr = [\"activo\", \"roa\", \"roe\", \"eficiencia\", \"cartera_irregular\",\n"
        "             \"tea_personal_mediana\", \"tea_tarjeta_mediana\",\n"
        "             \"comision_tarjeta_mediana\", \"ingreso_min_personal\", \"variedad_total\"]\n"
        "corr_of = cruce[cols_corr].corr(method=\"spearman\")\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(10, 8))\n"
        "sns.heatmap(corr_of, annot=True, fmt=\".2f\", cmap=\"RdBu_r\", center=0,\n"
        "            square=True, vmin=-1, vmax=1, ax=ax)\n"
        "ax.set_title(\"Correlación Spearman — balance vs oferta comercial (Dic-2024)\")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    code(
        "# Scatters ilustrativos\n"
        "fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n"
        "sns.scatterplot(data=cruce, x=\"activo\", y=\"tea_personal_mediana\",\n"
        "                hue=\"tipo_entidad\", palette=PALETA, hue_order=ORDEN_TIPO,\n"
        "                s=80, ax=axes[0])\n"
        "axes[0].set_xscale(\"log\")\n"
        "axes[0].set_title(\"Tamaño vs TEA personal\")\n"
        "axes[0].legend(fontsize=8)\n"
        "\n"
        "sns.scatterplot(data=cruce, x=\"roa\", y=\"tea_tarjeta_mediana\",\n"
        "                hue=\"tipo_entidad\", palette=PALETA, hue_order=ORDEN_TIPO,\n"
        "                s=80, ax=axes[1])\n"
        "axes[1].set_title(\"ROA vs TEA tarjeta\")\n"
        "axes[1].legend(fontsize=8)\n"
        "\n"
        "sns.scatterplot(data=cruce, x=\"activo\", y=\"variedad_total\",\n"
        "                hue=\"tipo_entidad\", palette=PALETA, hue_order=ORDEN_TIPO,\n"
        "                s=80, ax=axes[2])\n"
        "axes[2].set_xscale(\"log\")\n"
        "axes[2].set_title(\"Tamaño vs variedad de oferta\")\n"
        "axes[2].legend(fontsize=8)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "### 7.7 Takeaways de oferta comercial\n"
        "\n"
        "Los datos quedan guardados en `oferta_banco` (agregado a nivel banco) y `variedad` (variedad de productos) "
        "para ser integrados al panel del notebook 03 de clustering."
    ),
]

# Encuentro el índice de la sección 6 "## 6. Artefactos" para insertar antes.
idx_sec6 = None
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if cell["cell_type"] == "markdown" and src.startswith("## 6. Artefactos"):
        idx_sec6 = i
        break

if idx_sec6 is None:
    raise SystemExit("No encontré la sección 6 en el notebook")

nb["cells"] = nb["cells"][:idx_sec6] + nuevas + nb["cells"][idx_sec6:]

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"Insertadas {len(nuevas)} celdas antes del índice {idx_sec6}")
print(f"Total celdas ahora: {len(nb['cells'])}")
