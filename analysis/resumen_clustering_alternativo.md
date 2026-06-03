# Resumen comparativo — Clustering alternativo

Este documento compara los tres modelos de clustering desarrollados (notebooks 03, 04 y 05) y argumenta la elección del modelo final para la tesis.

## 1. Motivación de la exploración

El notebook 03 (modelo base, K-Means k=3 sobre 21 features con StandardScaler) alcanzó un silhouette de 0.243, una estructura débil pero interpretable. Para evaluar si se podía mejorar la calidad del agrupamiento se exploraron dos alternativas metodológicas:

- **Notebook 04**: log-transform sobre variables sesgadas + RobustScaler + GMM, sobre las 10 variables continuas (sin binarias de oferta).
- **Notebook 05**: espacio mixto que combina las 10 continuas transformadas (como en 04) con las 11 binarias de oferta concatenadas (como en 03), evaluando dos ponderaciones (peso 1.0 y peso 0.5 para las binarias).

## 2. Resultados comparativos

| Notebook | Modelo | Espacio | Silhouette | Bancos digitales agrupados | ARI vs nb 03 |
|---|---|---|---|---|---|
| 03 | K-Means k=3 | 21 feat + StandardScaler | **0.243** | No (dispersos en 3 clusters) | — |
| 04 | GMM k=3 | 10 cont + log + RobustScaler | **0.262** | Sí (cluster propio) | 0.215 |
| 05A | GMM k=3 | mixto (peso binarias = 1.0) | 0.145 | Sí (dentro del cluster chicos) | **0.594** |
| 05B | GMM k=3 | mixto (peso binarias = 0.5) | 0.174 | Sí (dentro del cluster chicos) | 0.427 |
| 05A | K-Means k=3 | mixto (peso 1.0) | 0.219 | No (singleton: outlier solo) | — |
| 05B | K-Means k=3 | mixto (peso 0.5) | 0.249 | No (singleton: outlier solo) | — |

## 3. Lectura cualitativa de cada modelo

### Notebook 03 (modelo base) — silhouette 0.243

Tres perfiles bien diferenciados a nivel de balance:

- **Cluster 1 (11 bancos)**: mayoristas/inversión chicos. Banco de Valores, Mariva, BICE, BACS, Citibank.
- **Cluster 2 (29 bancos)**: minoristas de gran escala. Galicia, Macro, BBVA, todos los públicos provinciales + Nación.
- **Cluster 3 (12 bancos)**: chicos con mora alta. ROA -1.4%, cartera irregular 9.4%.

**Limitación**: los 4 bancos 100% digitales (Brubank, Uala, Voii, Dino) quedan repartidos entre los tres clusters. La estructura "fintech" no aparece.

### Notebook 04 (GMM sobre continuas) — silhouette 0.262

El log + RobustScaler comprimió los outliers pero perdió la dimensión de oferta. PC1 pasó de explicar 26% a 47%, dominado por ROA. El clustering quedó así:

- **Cluster 1 (4 bancos)**: ¡todos los digitales agrupados! Brubank, Uala, Voii, Dino.
- **Cluster 2 (40 bancos)**: cluster gigante con casi todo lo demás.
- **Cluster 3 (8 bancos)**: bancos chicos con ROA negativo y eficiencia alta.

**Hallazgo positivo**: identificó por primera vez el clúster fintech.
**Limitación**: ARI con el 03 = 0.215 → la partición es esencialmente otra; se perdió la distinción mayorista vs minorista grande que el 03 sí veía.

### Notebook 05A (espacio mixto, peso 1.0) — silhouette 0.145

Recupera la oferta de productos y restaura la distinción mayorista/minorista. Los perfiles son los más interpretables de los tres modelos:

- **Cluster 1 — Minoristas masivos (27 bancos)**: ROA 3.1%, eficiencia 43%, depósitos 66% del activo. Ofrecen hipotecas (78%), paquetes premium (67%), tarjetas premium (59%). Incluye a Galicia, Nación, BBVA, Santander, todos los públicos provinciales, Macro, Patagonia, Supervielle, Credicoop. Es el núcleo del sistema bancario tradicional.

- **Cluster 2 — Chicos minoristas en transformación (15 bancos)**: ROA -2.3%, eficiencia 80%, cartera irregular 4.3%. Ofrecen personales (87%) pero casi no hipotecas (20%). Incluye **a los 4 digitales (Brubank, Uala, Voii, Dino)** junto con bancos de consumo (Servicios Financieros, Columbia, Sucrédito, Masventas) y privados chicos con dificultades (Meridiano, Julio, Saenz). El sub-perfil fintech queda anidado dentro de un cluster más amplio que comparte el rasgo "chicos minoristas con productos consumo y rentabilidad débil".

- **Cluster 3 — Mayoristas/inversión (10 bancos)**: ROA 3.2%, eficiencia 30%, cartera irregular 0.6%, depósitos solo 45% del activo. **Casi no ofrecen productos minoristas** (hipotecas 10%, paquetes 10%, personales 30%). Incluye Citibank, Banco de Valores, Mariva, BICE, CMF, BACS, Servicios y Transacciones, Comercio. Banca corporativa y de capital de mercado.

**Comparación con el 03**: ARI 0.594 → es la versión que más se parece al modelo base entre los tres, pero con mejor interpretación de las fronteras.

### Notebook 05B (espacio mixto, peso 0.5) — silhouette 0.174

Variante intermedia. Mantiene la estructura general pero el Cluster 3 (mayoristas) pierde definición (algunos se mezclan con minoristas). ARI con 03 = 0.427, más bajo que 05A. Los perfiles de oferta tienen rango más comprimido. **No aporta sobre 05A.**

## 4. ¿Por qué el silhouette bajó en el 05A?

El silhouette mide compactness geométrica: cuán cerca está cada banco de su cluster y cuán lejos de los otros. En el espacio mixto:

- Las 11 binarias agregan dimensionalidad efectiva. Aunque las binarias tienen rango fijo [0, 1] y los algoritmos las manejan, **agregar dimensiones siempre tiende a reducir el silhouette** (la "maldición de la dimensionalidad").
- La banca minorista comparte muchas binarias con la chica (ambas ofrecen plazo fijo, personal, caja). Los perfiles **se diferencian por combinaciones específicas** (paquete + hipoteca + premium), no por una variable que separe nítidamente.
- El silhouette castiga el solapamiento; los perfiles del 05A son **gradientes** (un banco puede tener oferta de paquete pero no premium, o hipoteca tradicional pero no UVA), y eso reduce la métrica.

**Pero un silhouette más bajo no implica peor clustering** cuando:

1. La partición es más interpretable (lo es: tres modelos de negocio claros).
2. Refleja la heterogeneidad real del sistema (la lo hace: ARI 0.594 con el 03 valida la estructura).
3. Recupera información que el 04 había perdido (lo hace: la dimensión de oferta vuelve a los primeros componentes).

## 5. Recomendación: notebook 05A como modelo principal

El **notebook 05A (GMM k=3 sobre espacio mixto, peso 1.0)** se propone como modelo principal de la tesis por estas razones:

1. **Interpretabilidad sustantiva**: los tres clusters mapean a tres modelos de negocio claros (minorista masivo / chicos en transformación / mayorista-inversión) que se explican por la combinación de balance + oferta de productos.
2. **Recuperación de la estructura del 03**: ARI 0.594 muestra que valida la intuición del modelo base, pero con fronteras mejor definidas (el cluster 3 mayorista pasa de 11 bancos en el 03 a 10 bancos más nítidos en el 05A).
3. **Captura del fenómeno fintech**: identifica el sub-grupo de digitales (Brubank, Uala, Voii, Dino) dentro del cluster 2, en lugar de dispersarlo como el 03.
4. **Robustez del algoritmo**: GMM maneja el outlier (Banco de Servicios Financieros) integrándolo al Cluster 2 con probabilidad 1.00, sin necesidad de exclusión manual.
5. **Comparabilidad temporal futura**: GMM da probabilidades de pertenencia, lo que permitirá analizar transiciones entre cortes (Dic-23 → Dic-25) con mayor granularidad que K-Means.

### El silhouette como métrica complementaria, no decisoria

El silhouette de 0.243 (nb 03) y 0.145 (nb 05A) ambos están **muy por debajo del umbral 0.50** que indica estructura fuerte. Esto es información sustantiva: el sistema bancario argentino no se separa en clusters geométricamente compactos porque hay **continuos genuinos** entre los perfiles (Banco BICA está entre minorista grande y chico; BancoSol está entre minorista chico y mayorista). Cualquier partición discreta va a tener silhouette bajo. La elección entre modelos debe basarse en **interpretabilidad y validez sustantiva**, no en la métrica.

## 6. Hallazgos transversales de la exploración

Independientemente del modelo elegido, los tres notebooks coinciden en estos puntos:

1. **El tipo de entidad (público/privado/extranjero) no segmenta a los bancos.** Los públicos provinciales y los privados grandes comparten cluster cuando comparten modelo de negocio (minorista masivo).
2. **Banco de Servicios Financieros es un outlier estructural.** Su balance (liquidez 1622%, depósitos 0.3%) refleja un modelo de consumer finance puro. Cualquier algoritmo lo aísla o lo coloca en un cluster donde claramente no encaja.
3. **El sistema no es jerárquico ni denso.** El cofenético del notebook 03 (0.457) y los 84% de ruido en DBSCAN confirman que no hay sub-estructura limpia más allá del nivel k=3.
4. **Los bancos 100% digitales son un fenómeno emergente identificable.** Brubank, Uala, Voii y Dino se agrupan consistentemente en el 04 y el 05; el 03 los disperaba porque las binarias de oferta no lograban diferenciarlos sin las transformaciones robustas del balance.

## 7. Próximos pasos sugeridos

- Adoptar el **05A como modelo principal** y reescribir la sección 7 del notebook 03 (interpretación) ya en el marco del 05A.
- Actualizar `analysis/resumen_clustering.md` con las conclusiones del 05A en lugar del 03.
- Para la entrega final de la tesis: presentar el 05A como modelo y los notebooks 03 y 04 como **anexos metodológicos** que documentan el proceso de selección.
- Analizar transiciones de cluster entre Dic-23 y Dic-25 usando las probabilidades del GMM (¿algún banco "migró" entre perfiles? ¿se acelera la digitalización?).
- Incorporar las tasas y comisiones de `oferta_banco.csv` como continuas adicionales en una iteración futura.
