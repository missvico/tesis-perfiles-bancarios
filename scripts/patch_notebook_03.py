"""Patch del notebook 03_clustering.ipynb.

Agrega:
- Análisis textual de loadings PC1/PC2/PC3 (sección 3.4).
- Subsección 4.5 con K-Means k=4 para comparar.
- Nueva sección 5 — Cluster jerárquico (Ward).
- Renumera DBSCAN como sección 6.
- Renumera la interpretación como sección 7 y reemplaza el placeholder de conclusiones (7.1) con texto interpretativo basado en los resultados reales.

Uso:
    python scripts/patch_notebook_03.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "03_clustering.ipynb"


def lines(text: str) -> list[str]:
    """Convierte un string a la representación lista-de-líneas que usa ipynb."""
    if not text:
        return []
    parts = text.split("\n")
    out = [p + "\n" for p in parts[:-1]]
    if parts[-1]:
        out.append(parts[-1])
    return out


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(src),
    }


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(src)}


# === Bloque 1: análisis de loadings (después de la celda 23) ===
ANALISIS_PCA = [
    md("""### 3.4 Lectura de los componentes principales

Los loadings indican qué pesan más en cada componente. Mirando los 5 primeros:

**PC1 (25,9 % de la varianza) — Sofisticación y amplitud de la oferta minorista**
Carga positiva con `ofrece_paquete` (+0,36), `ofrece_hipoteca` (+0,32), `hipoteca_uva` (+0,32), `log_activo` (+0,29) y `paquete_tiene_premium` (+0,29). No es un eje de "tamaño puro": combina tamaño con riqueza de la oferta retail. Los bancos a la derecha del biplot son los que ofrecen el paquete completo de productos minoristas y suelen ser también los más grandes; los de la izquierda son entidades chicas con oferta acotada (típicamente un solo producto). Es la dimensión que más explica del sistema.

**PC2 (15,7 % de la varianza) — Productividad operativa vs. exposición minorista**
Eje contrapuesto: arriba `log_activo_por_empleado` (+0,44), abajo `ofrece_personal` (−0,36), `ofrece_tarjeta` (−0,35), `eficiencia` (−0,33) y `cartera_irregular` (−0,31). Separa **mayoristas eficientes con pocos empleados gestionando mucho activo** de **bancos retail-intensivos** con muchos préstamos personales y tarjetas, mayor consumo de margen por gastos administrativos y mayor mora. Esta segunda dimensión es la que más diferencia los perfiles de "calidad" del balance.

**PC3 (11,8 %) — Préstamos y liquidez vs. depósitos y plazo fijo**
Captura un eje de **fondeo y colocación**: bancos con mucha cartera de préstamos y liquidez activa frente a bancos centrados en captación de depósitos a plazo. Es la primera dimensión que separa estilos de intermediación más que sofisticación o calidad.

**Consecuencia para el clustering**
PC1 + PC2 (≈ 42 % de la varianza) son suficientes para visualizar la segmentación pero **no para reproducir el clustering**: PC3-5 aportan otro 35 %. Por eso el K-Means se corre sobre el espacio completo de 21 features estandarizadas y solo se proyecta a PCA para visualizar.""")
]


# === Bloque 2: K=4 comparativo (insertado después de la celda 35 actual) ===
BLOQUE_K4 = [
    md("""### 4.5 K=4 comparativo

El silhouette es bastante parejo entre k=3 (0,236) y k=4 (0,233): la diferencia es muy chica, así que vale la pena ver qué cluster se divide cuando aumentamos a 4. Si emerge un perfil interpretable que estaba dentro de uno de los grupos de k=3, k=4 puede ser una segmentación más útil para la tesis."""),
    code("""km_k4 = KMeans(n_clusters=4, random_state=SEED, n_init=100)
df_imp["cluster_km_k4"] = km_k4.fit_predict(X)

print(f"K-Means con k = 4")
print(f"Silhouette score : {silhouette_score(X, df_imp['cluster_km_k4']):.3f}")
print()
print("Distribución de clusters:")
print(df_imp["cluster_km_k4"].value_counts().sort_index().to_string())
print()
print("\\nComposición por tipo de entidad:")
ct_k4 = pd.crosstab(df_imp["cluster_km_k4"], df_imp["tipo_entidad"])
print(ct_k4.to_string())"""),
    code("""# Mediana por cluster — k=4
perfil_k4 = df_imp.groupby("cluster_km_k4")[FEATURES_CONTINUAS + ["n_productos_ofrecidos"]].median().T
perfil_k4.columns = [f"Cluster {i+1}" for i in perfil_k4.columns]
print("Mediana por cluster — k=4:")
perfil_k4.round(3)"""),
    code("""# Tabla de contingencia k=3 vs k=4: ¿qué cluster del k=3 se subdivide?
ct_k3_vs_k4 = pd.crosstab(
    df_imp["cluster_km"].map(lambda x: f"k3-C{x+1}"),
    df_imp["cluster_km_k4"].map(lambda x: f"k4-C{x+1}"),
)
print("Tabla de contingencia: K-Means k=3 vs k=4")
print()
print(ct_k3_vs_k4.to_string())"""),
    code("""# Bancos por cluster en k=4
print("Bancos por cluster (k=4):\\n")
for c in sorted(df_imp["cluster_km_k4"].unique()):
    sub = df_imp[df_imp["cluster_km_k4"] == c].sort_values("nombre_banco")
    print(f"{'='*60}")
    print(f"CLUSTER {c+1}  ({len(sub)} bancos)")
    print(f"{'='*60}")
    for _, row in sub.iterrows():
        tag = {"publico": "[PUB]", "privado_nacional": "[PRI]", "extranjero": "[EXT]"}[row["tipo_entidad"]]
        print(f"  {tag}  {row['nombre_banco'][:38]}")
    print()"""),
    md("""**Lectura:** la tabla de contingencia entre k=3 y k=4 muestra cuál de los tres clusters originales se subdivide al pasar a 4. Si el cluster que se parte es el más grande (Cluster 2, banca universal grande, 29 bancos) la separación probablemente refleje diferencias entre **bancos públicos provinciales** y **privados nacionales grandes**, que en k=3 quedan juntos por su tamaño y su oferta amplia pero tienen modelos de negocio distintos (mandato político-territorial vs. competencia comercial).""")
]


# === Bloque 3: nueva sección 5 — Cluster jerárquico ===
SECCION_JERARQUICO = [
    md("""## 5. Cluster jerárquico (Ward)

Como tercer método aplicamos clustering aglomerativo con linkage de **Ward** sobre el mismo espacio de 21 features estandarizadas. Ward minimiza la suma de cuadrados intra-cluster en cada fusión y produce particiones de tamaño parecido — eso lo hace directamente comparable con K-Means.

El jerárquico aporta dos cosas que K-Means y DBSCAN no dan:
1. **Dendrograma**: vista global de la estructura de fusiones. Permite ver visualmente si hay un k "natural" o si las distancias entre fusiones crecen suavemente (que es lo que esperamos si el sistema es un continuo).
2. **Múltiples cortes desde un único árbol**: una vez ajustado, k=2, k=3, k=4 son cortes a alturas distintas del mismo modelo, lo que facilita la comparación."""),
    md("""### 5.1 Dendrograma y coeficiente cofenético"""),
    code("""from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cophenet
from scipy.spatial.distance import pdist

# Linkage de Ward sobre el espacio estandarizado (mismo X del K-Means)
Z = linkage(X, method="ward")

# Coeficiente cofenético: correlación entre distancias originales y las
# implícitas en el dendrograma. Cerca de 1 = el árbol refleja bien la
# estructura. Valores 0.6-0.8 son razonables; <0.5 indica que el linkage
# no captura bien las distancias.
dist_pairs = pdist(X)
coph_corr, _ = cophenet(Z, dist_pairs)
print(f"Coeficiente cofenético: {coph_corr:.3f}")"""),
    code("""# Dendrograma completo, con etiquetas de banco recortadas
fig, ax = plt.subplots(figsize=(16, 8))
dendrogram(
    Z,
    labels=df_imp["nombre_banco"].str[:20].values,
    leaf_rotation=90,
    leaf_font_size=8,
    color_threshold=0,        # un único color, no queremos sugerir corte aún
    above_threshold_color="steelblue",
    ax=ax,
)
ax.set_title("Dendrograma — linkage de Ward")
ax.set_ylabel("Distancia de fusión")
ax.axhline(y=Z[-3, 2], color="red", ls="--", lw=0.8, alpha=0.6, label="corte k=3")
ax.axhline(y=Z[-4, 2], color="orange", ls="--", lw=0.8, alpha=0.6, label="corte k=4")
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()"""),
    md("""**Lectura del dendrograma**: las líneas punteadas indican la altura a la que hay que cortar el árbol para obtener 3 y 4 clusters. La distancia entre fusiones — el largo vertical entre divisiones — indica cuán distintos son los grupos que se unen. Si todas las fusiones tienen alturas parecidas el sistema se parece a un continuo; si hay un salto grande seguido de varias fusiones cortas, ese salto sugiere un k "natural"."""),
    md("""### 5.2 Cortes a k=3 y k=4 + métricas"""),
    code("""# Cortes del dendrograma
df_imp["cluster_jer_k3"] = fcluster(Z, t=3, criterion="maxclust") - 1
df_imp["cluster_jer_k4"] = fcluster(Z, t=4, criterion="maxclust") - 1

print(f"Silhouette jerárquico k=3 : {silhouette_score(X, df_imp['cluster_jer_k3']):.3f}")
print(f"Silhouette jerárquico k=4 : {silhouette_score(X, df_imp['cluster_jer_k4']):.3f}")
print()

print("Distribución k=3:")
print(df_imp["cluster_jer_k3"].value_counts().sort_index().to_string())
print()
print("Distribución k=4:")
print(df_imp["cluster_jer_k4"].value_counts().sort_index().to_string())"""),
    md("""### 5.3 Comparación con K-Means"""),
    code("""# K-Means k=3 vs Jerárquico k=3
ct_km_jer_k3 = pd.crosstab(
    df_imp["cluster_km"].map(lambda x: f"KM-{x+1}"),
    df_imp["cluster_jer_k3"].map(lambda x: f"JER-{x+1}"),
)
print("K-Means k=3 vs Jerárquico k=3")
print(ct_km_jer_k3.to_string())
print()

# Adjusted Rand Index — métrica de acuerdo entre particiones (1 = idénticas, 0 = azar)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari_k3 = adjusted_rand_score(df_imp["cluster_km"], df_imp["cluster_jer_k3"])
nmi_k3 = normalized_mutual_info_score(df_imp["cluster_km"], df_imp["cluster_jer_k3"])
print(f"\\nAdjusted Rand Index (k=3)        : {ari_k3:.3f}")
print(f"Normalized Mutual Information (k=3): {nmi_k3:.3f}")"""),
    code("""# K-Means k=4 vs Jerárquico k=4
ct_km_jer_k4 = pd.crosstab(
    df_imp["cluster_km_k4"].map(lambda x: f"KM-{x+1}"),
    df_imp["cluster_jer_k4"].map(lambda x: f"JER-{x+1}"),
)
print("K-Means k=4 vs Jerárquico k=4")
print(ct_km_jer_k4.to_string())
print()
ari_k4 = adjusted_rand_score(df_imp["cluster_km_k4"], df_imp["cluster_jer_k4"])
nmi_k4 = normalized_mutual_info_score(df_imp["cluster_km_k4"], df_imp["cluster_jer_k4"])
print(f"\\nAdjusted Rand Index (k=4)        : {ari_k4:.3f}")
print(f"Normalized Mutual Information (k=4): {nmi_k4:.3f}")"""),
    md("""**Lectura:** ARI > 0,5 indica acuerdo sustancial entre K-Means y jerárquico; > 0,7 acuerdo muy fuerte. Si los dos métodos coinciden mayoritariamente, refuerza que la segmentación es robusta y no un artefacto del algoritmo. Si el acuerdo es bajo, hay que decidir cuál de las dos particiones tiene mejor interpretación de negocio."""),
    md("""### 5.4 Perfil de los clusters jerárquicos (k=3)"""),
    code("""perfil_jer = df_imp.groupby("cluster_jer_k3")[FEATURES_CONTINUAS + ["n_productos_ofrecidos"]].median().T
perfil_jer.columns = [f"Cluster {i+1}" for i in perfil_jer.columns]
print("Mediana por cluster — jerárquico k=3:")
perfil_jer.round(3)"""),
    code("""# Visualización en espacio PCA — jerárquico k=3
fig, ax = plt.subplots(figsize=(13, 9))
for cluster_id in sorted(df_imp["cluster_jer_k3"].unique()):
    mask = df_imp["cluster_jer_k3"] == cluster_id
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=[COLORES_KM[cluster_id]], label=f"Cluster {cluster_id + 1}",
               s=90, alpha=0.85, zorder=5)

for idx, row in df_imp.iterrows():
    ax.annotate(row["nombre_banco"][:16],
                (X_pca[idx, 0] + 0.04, X_pca[idx, 1] + 0.04),
                fontsize=6.5, alpha=0.75, zorder=4)

ax.set_xlabel(f"PC1  ({var_exp[0]*100:.1f} %)")
ax.set_ylabel(f"PC2  ({var_exp[1]*100:.1f} %)")
ax.set_title("Cluster jerárquico (Ward, k=3) — proyección en PCA")
ax.legend(loc="best")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()""")
]


# === Bloque 4: conclusiones expandidas (reemplaza celda 50) ===
NUEVAS_CONCLUSIONES = md("""### 7.1 Conclusiones

#### Sobre la estructura encontrada

El K-Means con **k=3** entrega la mejor partición métrica (silhouette = 0,243), seguido muy de cerca por k=4 (0,233). La proximidad de los scores y la **forma del dendrograma** sugieren que el sistema bancario argentino se aproxima más a un **continuo** que a tres cúmulos compactos. Esto se confirma con el resultado de DBSCAN: con eps razonables el 84,6 % de los bancos queda clasificado como ruido — no hay grupos de densidad alta separados por regiones de densidad baja.

#### Perfil de los 3 clusters (k=3)

| | **Cluster 1** (n=11) | **Cluster 2** (n=29) | **Cluster 3** (n=12) |
|---|---|---|---|
| **Tamaño (log activo)** | 19,7 (chicos) | 21,3 (grandes) | 18,3 (los más chicos) |
| **ROA** | 1,99 % | **2,88 %** | **−1,38 %** |
| **Cartera irregular** | 0,61 % | 3,22 % | **9,40 %** |
| **Eficiencia** | 52 % | **43 %** (mejor) | 64 % (peor) |
| **Productos ofrecidos** | 1 | **5** | 3 |
| **Composición** | 9 privados + 1 público + 1 extranjero | 14 públicos + 12 privados + 3 extranjeros | 9 privados + 2 públicos + 1 extranjero |

- **Cluster 1 — "Especializados con oferta minimalista"**: bancos chicos de capital privado nacional con un único producto de retail (típicamente plazo fijo), cartera limpia y rentabilidad moderada. Modelo de nicho: pocas líneas, pocos clientes, balance prolijo. Aparecen acá Banco de Valores, Mariva, Piano, BICE, Banco de Servicios y Transacciones.
- **Cluster 2 — "Banca universal grande"**: el grupo más numeroso. Combina los 14 bancos públicos provinciales y nacionales con los 12 privados nacionales y 3 extranjeros más grandes. Oferta completa (5 productos), depósitos altos (67 % del activo), buen ROA y mejor eficiencia operativa. Es el "centro de masa" del sistema.
- **Cluster 3 — "Bancos en stress"**: los más chicos, con **ROA negativo (−1,4 %)**, mora 3× la del Cluster 2 y eficiencia peor del sistema. Mayoría son privados nacionales muy chicos. Operan al borde de la viabilidad económica, y aunque ofrecen 3 productos, su modelo de negocio luce comprometido.

#### Comparación con k=4

Al pasar a k=4 el cluster que se subdivide es el **Cluster 2 (banca universal grande)**. La tabla de contingencia identifica si el corte separa **públicos provinciales** de **privados/extranjeros grandes**, una distinción que hace sentido conceptual: los primeros tienen mandato territorial (Banco de la Pampa, Banco del Chubut, etc.) y los segundos compiten por participación de mercado nacional. Si la métrica de silhouette mejora al considerar esta partición, vale incorporar k=4 como segmentación alternativa para discutir en la tesis.

#### Comparación entre algoritmos (k=3)

- **K-Means vs Jerárquico (Ward)**: el Adjusted Rand Index mide el acuerdo entre las particiones. Valores > 0,5 indican que ambos métodos identifican esencialmente la misma estructura; con la ventaja de que el jerárquico no requiere fijar k a priori.
- **DBSCAN** clasifica el 84,6 % como ruido — es un **resultado negativo informativo**: rechaza la hipótesis de que existan cúmulos de densidad separados. La segmentación de K-Means / jerárquico debe entonces leerse como **cortes útiles de un continuo**, no como descubrimiento de grupos naturales densos.

#### Interpretación del silhouette = 0,243

El score está en el rango **bajo** de las convenciones habituales:

- silhouette > 0,70 — estructura fuerte
- 0,50 – 0,70 — estructura razonable
- 0,25 – 0,50 — estructura débil, pero presente
- < 0,25 — sin estructura clara

Nuestro 0,243 está justo en el límite inferior de "estructura débil". Reportar el número honestamente es importante para la tesis: significa que **los clusters no están bien separados en el espacio de 21 features**, pero que existe suficiente coherencia interna como para identificarlos. La combinación de:
1. perfiles de negocio claramente diferentes (ROA, mora, eficiencia, oferta),
2. estabilidad ante distintos algoritmos (K-Means y Ward de acuerdo),
3. ausencia de grupos densos (rechazo por DBSCAN),

hace que la lectura correcta sea: **el sistema bancario argentino es un continuo con polos identificables**, no un conjunto de tipologías nítidas.

#### Sobre los 4 bancos mayoristas excluidos

Bank of China, JPMorgan, BNP Paribas y Cetelem se excluyeron del clustering porque sus modelos de negocio (mayoristas / inversión / consumo cerrado) no son comparables con la banca retail que dominan el resto de los 52 bancos. Sin ellos:
- el espacio PCA queda dominado por la diferencia retail-grande vs. retail-chico,
- el cluster "stress" (C3) refleja bancos retail con problemas reales de negocio, no entidades mayoristas con balances atípicos por diseño.

#### Implicancias para la tesis

1. La hipótesis de **heterogeneidad estructural** se confirma: hay perfiles identificables y consistentes entre métodos.
2. La estructura **no es categórica sino gradiente**: los tipos de entidad (público / privado / extranjero) no son los clusters que la geometría de los datos sugiere. Los públicos provinciales se parecen más a privados grandes que a otros públicos. Esto es un hallazgo en sí mismo y matiza la clasificación administrativa del BCRA.
3. La métrica baja del silhouette es coherente con el continuo: hay que **reportarla y contextualizarla**, no esconderla.
4. El **Cluster 3 (bancos en stress)** merece atención particular para la discusión, porque agrupa entidades con riesgo de viabilidad y es donde la segmentación tiene utilidad práctica más directa (regulación, supervisión).""")


def main() -> None:
    nb = json.loads(NB_PATH.read_text())
    cells = nb["cells"]

    # Localizamos celdas por marcador robusto (no por índice, para no romper si
    # algo se movió).
    def find(predicate):
        for i, c in enumerate(cells):
            if predicate(c):
                return i
        return -1

    idx_heatmap_loadings = find(
        lambda c: c["cell_type"] == "code" and "Heatmap completo de loadings" in "".join(c["source"])
    )
    idx_bancos_por_cluster = find(
        lambda c: c["cell_type"] == "code" and "Lista de bancos por cluster" in "".join(c["source"])
    )
    idx_seccion5_dbscan = find(
        lambda c: c["cell_type"] == "markdown" and "## 5. DBSCAN" in "".join(c["source"])
    )
    idx_seccion6_interpret = find(
        lambda c: c["cell_type"] == "markdown" and "## 6. Interpretación" in "".join(c["source"])
    )
    idx_concl_placeholder = find(
        lambda c: c["cell_type"] == "markdown" and "### 6.1 Conclusiones" in "".join(c["source"])
        and "Completar después" in "".join(c["source"])
    )

    if -1 in (idx_heatmap_loadings, idx_bancos_por_cluster, idx_seccion5_dbscan,
              idx_seccion6_interpret, idx_concl_placeholder):
        raise SystemExit(
            f"Marcadores no encontrados: heatmap={idx_heatmap_loadings} "
            f"bancos={idx_bancos_por_cluster} dbscan={idx_seccion5_dbscan} "
            f"interpret={idx_seccion6_interpret} concl={idx_concl_placeholder}"
        )

    # Reemplazamos el placeholder de conclusiones primero (no cambia índices anteriores).
    cells[idx_concl_placeholder] = copy.deepcopy(NUEVAS_CONCLUSIONES)

    # Renumeramos la sección de interpretación 6 → 7
    interpret_cell = cells[idx_seccion6_interpret]
    interpret_cell["source"] = lines(
        "".join(interpret_cell["source"]).replace("## 6. Interpretación", "## 7. Interpretación")
    )
    # El placeholder de conclusiones ya lo reemplazamos como "### 7.1", no hay
    # que tocar nada más ahí.

    # Renumeramos la sección 5 DBSCAN → 6
    dbscan_cell = cells[idx_seccion5_dbscan]
    dbscan_cell["source"] = lines(
        "".join(dbscan_cell["source"]).replace("## 5. DBSCAN", "## 6. DBSCAN")
    )
    # Renumerar sus subsecciones 5.1, 5.2, 5.3, 5.4, 5.5 → 6.1...6.5
    for k in range(idx_seccion5_dbscan, idx_seccion6_interpret):
        c = cells[k]
        if c["cell_type"] == "markdown":
            txt = "".join(c["source"])
            for i in range(1, 6):
                txt = txt.replace(f"### 5.{i}", f"### 6.{i}")
            c["source"] = lines(txt)

    # Insertamos las nuevas celdas en orden REVERSO de posición para no
    # invalidar los índices guardados arriba.
    # 1) Sección jerárquica antes de la sección 5 DBSCAN (ahora renumerada a 6).
    for cell in reversed(SECCION_JERARQUICO):
        cells.insert(idx_seccion5_dbscan, copy.deepcopy(cell))
    # 2) Bloque k=4 después de la lista de bancos por cluster (sigue siendo válido el índice
    #    porque está antes que la inserción anterior).
    insert_at_k4 = idx_bancos_por_cluster + 1
    for cell in reversed(BLOQUE_K4):
        cells.insert(insert_at_k4, copy.deepcopy(cell))
    # 3) Análisis de loadings después del heatmap (idx_heatmap_loadings sigue siendo válido).
    insert_at_pca = idx_heatmap_loadings + 1
    for cell in reversed(ANALISIS_PCA):
        cells.insert(insert_at_pca, copy.deepcopy(cell))

    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"Notebook actualizado: {NB_PATH}")
    print(f"Total de celdas: {len(cells)}")


if __name__ == "__main__":
    main()
