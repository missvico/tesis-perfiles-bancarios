# Resumen de clustering — Entrega III

Este documento sintetiza los resultados del notebook `03_clustering.ipynb`. Acompaña a `resumen_eda.md` y constituye la base analítica de la Entrega III de la tesis.

## 1. Insumos y diseño del experimento

- **Panel**: promedio de los tres cortes (Dic-2023, Dic-2024, Dic-2025) del `panel_ratios.csv`, complementado con la oferta de productos por banco (`oferta_banco.csv`).
- **Universo de bancos**: se parte de 56 entidades y se excluyen 4 bancos mayoristas/extranjeros sin cartera minorista relevante (Bank of China, JPMorgan, BNP Paribas, Cetelem). Queda un panel de **52 bancos**.
- **Features (21)**: 10 variables continuas estandarizadas (`log_activo`, `prestamos_sobre_activo`, `titulos_sobre_activo`, `depositos_sobre_activo`, `patrimonio_sobre_activo`, `roa`, `liquidez`, `eficiencia`, `cartera_irregular`, `log_activo_por_empleado`, `n_productos_ofrecidos`) + 11 binarias de oferta (caja de ahorro pesos/USD, cuenta corriente, plazo fijo, tarjeta crédito/débito, préstamo personal/hipotecario/prendario, etc.).
- **Imputación**: missings en `cartera_irregular` y `n_productos_ofrecidos` se reemplazan por la mediana del panel (4 y 7 imputaciones respectivamente).
- **Estandarización**: StandardScaler antes de PCA y de todos los algoritmos.

## 2. PCA — interpretación de los ejes

Los primeros dos componentes explican **41.6 % de la varianza** (PC1 25.9 %, PC2 15.7 %). Se necesitan 7 componentes para llegar al 80 %, lo que indica que la heterogeneidad del sistema bancario argentino es genuinamente multidimensional y no se reduce a un único eje "tamaño".

- **PC1 — Sofisticación retail / escala de fondeo minorista**. Cargan positivo `depositos_sobre_activo`, `n_productos_ofrecidos`, las binarias de productos minoristas (caja de ahorro, plazo fijo, tarjeta de débito) y `log_activo`. Cargan negativo `patrimonio_sobre_activo` y `liquidez`. PC1 alto = bancos con fondeo masivo de depósitos minoristas y oferta de productos diversificada; PC1 bajo = bancos chicos, capitalizados y líquidos.
- **PC2 — Productividad operativa vs. exposición minorista**. Cargan positivo `log_activo_por_empleado` y `eficiencia`. Cargan negativo `prestamos_sobre_activo` y `cartera_irregular`. PC2 alto = banca corporativa/mayorista con alta productividad por empleado; PC2 bajo = banca minorista con mucha cartera y mayor mora.

Esta lectura empírica corrige el supuesto inicial de que PC1 captura "tamaño puro": el `log_activo` por sí solo no separa al sistema, lo hace en combinación con la oferta minorista.

## 3. K-Means k=3 (modelo base)

**Silhouette score: 0.243**. Estructura débil pero interpretable. La distribución es **11 / 29 / 12**.

| Cluster | Tamaño | Perfil mediano | Lectura |
|---|---|---|---|
| **1 — Mayoristas/inversión** | 11 | log_activo 19.7, depósitos 55%, eficiencia 52%, ROA 2.0%, cartera irregular 0.6%, n_productos 1 | Bancos chicos especializados (mercado de capitales, comercio exterior, securitización). Ejemplos: Banco de Valores, Mariva, BICE, BACS, Citibank. |
| **2 — Banca minorista de gran escala** | 29 | log_activo 21.3, depósitos 67%, préstamos 33%, ROA 2.9%, eficiencia 43%, n_productos 5 | El núcleo del sistema: bancos privados grandes (Galicia, Macro, BBVA, Santander), todos los públicos provinciales + Nación, y privados nacionales con red. |
| **3 — Chicos con cartera deteriorada** | 12 | log_activo 18.3, depósitos 47%, ROA −1.4%, eficiencia 64%, cartera irregular 9.4%, liquidez 25% | Bancos pequeños privados y públicos provinciales chicos con rentabilidad débil y mora alta. Incluye un outlier extremo (Banco de Servicios Financieros). |

**Observación clave**: el tipo de entidad (público/privado/extranjero) no segmenta a los bancos. El cluster 2 mezcla todos los bancos públicos provinciales con bancos privados grandes y filiales de extranjeros; lo que los une no es la propiedad sino el modelo de negocio (fondeo minorista de gran escala con oferta diversificada).

## 4. K=4 — ¿qué se gana subdividiendo?

**Silhouette baja a 0.204**, por debajo del óptimo k=3. La tabla de contingencia revela que el cuarto cluster **no subdivide al cluster minorista (C2)**, sino que aísla al outlier **Banco de Servicios Financieros**:

```
                k4-C1  k4-C2  k4-C3  k4-C4
cluster_km
k3-C1               8      0      3      0
k3-C2               0     23      6      0
k3-C3               0      0     11      1   ← el outlier aislado
```

El perfil del cluster 4 confirma el outlier: liquidez 1622 %, depósitos 0.3 %, préstamos 83 % del activo, ROA −4.5 %, cartera irregular 20.5 %. Es Banco de Servicios Financieros, una entidad de consumo (Frávega) con un balance atípico: prácticamente no toma depósitos y financia todo con capital propio.

**Conclusión metodológica**: k=4 no aporta una segmentación más rica; solo "descubre" un outlier que ya conocíamos. Para la tesis se mantiene k=3 como modelo principal.

## 5. K=3 sin el outlier — test de robustez

Repetimos K-Means k=3 excluyendo a Banco de Servicios Financieros (panel de 51 bancos).

- **Silhouette: 0.208** (vs. 0.243 con outlier) — la separación geométrica empeora levemente.
- **Adjusted Rand Index con la partición original: 0.539** — la partición **cambió de forma no trivial**.

Tabla de contingencia con vs. sin outlier:

```
                SIN-C1  SIN-C2  SIN-C3
CON-C1               0       3       8       (mayoristas)
CON-C2              23       6       0       (minoristas grandes)
CON-C3               0      11       0       (chicos con mora)
```

Lectura:

- Los **23 bancos minoristas grandes** (CON-C2) permanecen juntos en SIN-C1 → núcleo estable.
- Los **8 mayoristas más puros** (CON-C1) permanecen como SIN-C3.
- Los **11 chicos con mora** (CON-C3) se mezclan con 3 bancos que antes estaban en CON-C1 y 6 que estaban en CON-C2, dando lugar a un cluster intermedio SIN-C2 (medianos chicos con mora moderada de 4 %).

Esta inestabilidad refleja un fenómeno real: en el espacio de features, los bancos chicos forman un continuo y no un cluster bien definido. La presencia del outlier "tira" del centroide del cluster 3 hacia el extremo y disciplina la frontera; al sacarlo, esa frontera se recalibra. La estructura **macro** (minoristas vs. todo lo demás) es robusta; las **fronteras finas** dentro de "todo lo demás" no lo son.

## 6. Clustering jerárquico (Ward) — validación cruzada

Para chequear si la partición de K-Means es un artefacto del algoritmo, se aplica clustering aglomerativo con linkage Ward sobre los mismos features.

- **Coeficiente cofenético: 0.457** — correlación moderada entre las distancias del espacio original y las del dendrograma. Indica que el sistema bancario argentino **no es jerárquico** en el sentido fuerte: no hay subgrupos limpios anidados.
- **Silhouette jerárquico k=3: 0.175** (peor que K-Means 0.243). K-Means es preferible.
- **ARI K-Means vs. Ward (k=3): 0.446** — coincidencia moderada. Coinciden en los extremos (mayoristas y minoristas grandes) pero discrepan en el cluster intermedio. NMI 0.560.
- **A k=4**: ARI 0.539, NMI 0.640 — la coincidencia mejora porque ambos algoritmos aíslan al outlier.

**Lectura**: Ward y K-Means coinciden en el esqueleto (mayoristas vs. minoristas grandes vs. resto) pero difieren en los detalles, lo que confirma la conclusión de la sección 5: las fronteras finas no son robustas.

## 7. DBSCAN — el sistema no tiene "densidad" en el sentido clásico

Calibrado con `eps=1.5`, `min_samples=3`:

- **84.6 % de ruido** (44 de 52 bancos sin cluster). Solo se forman 2 clusters chicos de 4 bancos cada uno.
- Los bancos minoristas grandes (Galicia, BBVA, Santander, Macro, Patagonia, San Juan, Santa Cruz, Ciudad) son los únicos que forman regiones densas.

**Conclusión**: DBSCAN confirma que el sistema bancario argentino es disperso y no tiene grupos densamente compactos más allá del núcleo de minoristas grandes. K-Means impone una partición útil aunque artificial; DBSCAN se rehúsa a inventar grupos donde no los hay. No se usa como modelo principal pero sirve como diagnóstico: validar que muchos bancos chicos son **realmente atípicos** y no comparten densidad.

## 8. Síntesis para la tesis

1. **K-Means k=3 es el modelo principal**, con silhouette 0.243. Identifica tres perfiles de negocio: (i) banca de inversión/mayorista chica, (ii) banca minorista de gran escala y (iii) bancos chicos con cartera deteriorada.
2. **El tipo de entidad (propiedad) no estructura los clusters**. Bancos públicos provinciales y privados grandes comparten cluster cuando tienen el mismo modelo de negocio.
3. **La partición es estable en el núcleo y frágil en las fronteras**. Los 23 bancos minoristas grandes y los 8 mayoristas puros forman bloques sólidos; el cluster 3 (chicos con mora) es más una etiqueta residual que un grupo cohesionado.
4. **Banco de Servicios Financieros es un outlier estructural**, no un error de datos. Su balance refleja un modelo de negocio (consumer finance puro) que no tiene par en el sistema.
5. **El sistema no es jerárquico ni denso**: ni Ward ni DBSCAN encuentran sub-estructura limpia. Esto es información sustantiva: el sistema bancario argentino es disperso y multidimensional, y los tres clusters de K-Means son la mejor síntesis posible.

## 9. Limitaciones y próximos pasos

- El silhouette de 0.243 es bajo en términos absolutos. Implica solapamiento real, no falla del algoritmo: muchos bancos están "entre" perfiles.
- El promedio de los tres cortes oculta dinámica temporal. Una extensión natural es analizar transiciones de cluster entre Dic-23 → Dic-25.
- La oferta de productos contribuye con 11 features binarias pero su peso conjunto en PC1 podría merecer un análisis aparte (¿el modelo de negocio se está digitalizando entre cortes?).
- Las binarias de oferta capturan **disponibilidad**, no **volumen ni precio**. Un próximo paso sería incorporar las tasas y comisiones de `oferta_banco.csv` como variables continuas.
