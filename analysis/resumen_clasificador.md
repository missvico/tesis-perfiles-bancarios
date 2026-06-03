# Resumen — Clasificador supervisado LightGBM sobre el modelo 05A

Este documento sintetiza los resultados del notebook `06_clasificador_lgbm.ipynb`. Cierra el ciclo del análisis: el modelo de clustering principal (05A) es **validado** por un clasificador supervisado que aprende a predecir las etiquetas del GMM a partir de las features originales.

## 1. Motivación

El notebook 05 dejó al GMM sobre espacio mixto (peso binarias = 1.0) como modelo principal, con silhouette 0.145 — bajo en términos absolutos. El silhouette mide compactness geométrica y no responde a la pregunta sustantiva: **¿la partición es predecible? ¿tiene lógica más allá del algoritmo que la generó?**

Un clasificador supervisado responde esto. Si las 21 features originales permiten predecir la etiqueta del cluster con buena performance en CV, entonces:

1. **La partición es real**, no un artefacto del GMM: hay señal en los datos que define cada perfil.
2. Tenemos **interpretabilidad por feature y por clase** vía SHAP, complementando los perfiles medianos del 05A con información sobre qué variables y combinaciones definen cada cluster.
3. El modelo entrenado queda como **producto reusable** para clasificar nuevos bancos o cortes temporales futuros sin reentrenar el GMM.

## 2. Diseño experimental

- **Etiquetas**: cluster del GMM-A reproducido sobre los 52 bancos (mismo código que el notebook 05). Distribución 27 / 15 / 10.
- **Features**: 21 originales (10 continuas + 11 binarias) **sin transformar**. LightGBM es invariante a transformaciones monótonas; mantener escala original facilita la interpretación SHAP.
- **Evaluación honesta**: Nested Cross-Validation.
  - **Outer loop**: 5 folds estratificados. Métrica honesta de generalización.
  - **Inner loop**: 3 folds + `RandomizedSearchCV` con 30 iteraciones, optimizando **macro-F1** (no accuracy: las clases están desbalanceadas 27/15/10).
- **Modelo final** (para SHAP): reentrenado con `RandomizedSearchCV` (5 folds, 30 iter) sobre los 52 bancos. **No tiene métrica de generalización propia** — la métrica honesta sigue siendo la del outer loop.
- **Espacio de búsqueda**: `num_leaves` ∈ {7, 15, 31}, `max_depth` ∈ {3, 5, -1}, `learning_rate` ∈ {0.01, 0.05, 0.1}, `n_estimators` ∈ {100, 300, 500}, `min_child_samples` ∈ {2, 5, 10}, `reg_alpha` y `reg_lambda` ∈ {0, 0.1, 1.0}.
- **`class_weight="balanced"`** en LGBM para compensar el desbalanceo.

## 3. Resultados — Outer CV (métrica honesta)

| Fold | Accuracy | Macro-F1 | best F1 inner (no honesto) |
|---|---|---|---|
| 1 | 0.818 | 0.833 | 0.905 |
| 2 | 0.818 | 0.792 | 0.797 |
| 3 | 1.000 | 1.000 | 0.899 |
| 4 | 0.900 | 0.859 | 0.958 |
| 5 | 0.800 | 0.767 | 0.946 |
| **Promedio** | **0.867 ± 0.075** | **0.850 ± 0.081** | — |

**Por clase**:

| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| Minoristas masivos | 0.862 | 0.926 | 0.893 | 27 |
| Chicos transformación | 0.857 | 0.800 | 0.828 | 15 |
| Mayoristas/inversión | 0.889 | 0.800 | 0.842 | 10 |

**Lectura**: 0.867 de accuracy en CV anidado con n=52 supera nuestra franja "razonable" (0.65-0.80) y entra en "estructura fuerte" (> 0.85). Las tres clases tienen F1 entre 0.83 y 0.89 — el desbalance no degrada las clases chicas. La partición del 05A es **predecible y reproducible** desde las features individuales.

### Matriz de confusión (outer CV, n=52)

```
                       Pred: Min  Pred: Chic  Pred: May
Real: Minoristas         25         2          0
Real: Chicos             2          12         1
Real: Mayoristas         2          0          8
```

Errores totales: **7 / 52 = 13.5 %**. Mayoría son confusiones con "Minoristas masivos" — el cluster central que abarca el continuo más amplio.

## 4. Bancos mal clasificados — todos son "bordes" reconocibles

Los 7 errores no son aleatorios: son los bancos que ya el resumen comparativo del 05A había identificado como casos intermedios o atípicos.

| Banco | Real | Predicho | Por qué confunde |
|---|---|---|---|
| BROU (Rep. Oriental Uruguay) | Chicos transf. | Mayoristas | ROA -35 %, eficiencia 407 % (outlier extremo); depósitos 40 % lo acerca a mayoristas |
| Banco Columbia | Chicos transf. | Minoristas | ROA cercano a 0 + cartera irregular 10 % — perfil consumer finance ambiguo |
| Banco Dino | Chicos transf. | Minoristas | Digital con depósitos 60 % — el clasificador no usa la dimensión "digital" |
| Banco Sgo. del Estero | Mayoristas | Minoristas | Público provincial chico con depósitos 48 % — borde mayorista/minorista |
| Banco de Comercio | Mayoristas | Minoristas | ROA 6.4 %, préstamos 30 % — rentabilidad lo aleja del mayorista típico |
| Banco Municipal Rosario | Minoristas | Chicos transf. | Eficiencia 53 % + cartera irregular 2.4 % — borde minorista deteriorado |
| Banco Coinag | Minoristas | Chicos transf. | Eficiencia 62 % — borde inverso al anterior |

**Por qué esto valida la partición**: si los errores cayeran en bancos típicos (Galicia, BBVA, Nación), sería alarmante. Que sean justamente los del borde — los mismos que el GMM asignó con probabilidad más baja — confirma que GMM y LGBM están viendo **el mismo continuo subyacente**, simplemente lo cortan en lugares ligeramente distintos. Es exactamente el patrón esperado dado el silhouette bajo.

## 5. Estabilidad de hiperparámetros

Los 5 folds del outer loop convergen a configuraciones similares:

| Hiperparámetro | Valor modal | Folds que lo eligieron |
|---|---|---|
| `max_depth` | 3 | 3 / 5 |
| `min_child_samples` | 2 | 3 / 5 |
| `n_estimators` | 300 | 3 / 5 |
| `reg_alpha` | 0.0 | 3 / 5 |
| `num_leaves` | 15 ó 31 | 2 / 5 cada uno |
| `learning_rate` | 0.01 | 2 / 5 |
| `reg_lambda` | 1.0 | 2 / 5 |

Modelos chicos (max_depth 3, num_leaves bajos) sin mucha regularización — consistente con n=52. No hay folds eligiendo configuraciones extrañas, lo que sugiere que la señal es estable.

## 6. SHAP — qué define cada perfil

### 6.1 Importancia global (top features)

| Rank | Feature | |SHAP| promedio |
|---|---|---|
| 1 | `roa` | 1.319 |
| 2 | `cartera_irregular` | 0.852 |
| 3 | `eficiencia` | 0.655 |
| 4 | `ofrece_paquete` | 0.582 |
| 5 | `prestamos_sobre_activo` | 0.504 |
| 6 | `paquete_tiene_premium` | 0.455 |
| 7 | `log_activo` | 0.230 |
| 8 | `ofrece_hipoteca` | 0.228 |

Las 3 features dominantes son **financieras** (ROA, mora, eficiencia). Las binarias de oferta (`ofrece_paquete`, `paquete_tiene_premium`, `ofrece_hipoteca`) entran en el top y son las **dos categorías más discriminantes**: paquetes premium y crédito hipotecario.

Features que no aportan nada (SHAP ≈ 0): `ofrece_pfijo`, `ofrece_prendario`, `tarjeta_tiene_premium`. Son binarias casi constantes en el panel (lo ofrecen todos o casi nadie).

### 6.2 Importancia por clase

| Feature | Minoristas | Chicos transf. | Mayoristas |
|---|---|---|---|
| `roa` | 0.548 | **3.300** | 0.108 |
| `cartera_irregular` | 0.163 | 0.000 | **2.393** |
| `eficiencia` | 0.009 | **1.432** | 0.524 |
| `ofrece_paquete` | **1.567** | 0.039 | 0.139 |
| `prestamos_sobre_activo` | **1.147** | 0.000 | 0.366 |
| `paquete_tiene_premium` | **1.364** | 0.000 | 0.000 |
| `ofrece_hipoteca` | **0.684** | 0.000 | 0.000 |
| `log_activo` | 0.486 | 0.196 | 0.009 |

**Cada cluster tiene una "firma" distinta**:

- **Minoristas masivos**: el clasificador los identifica por **oferta retail premium** (paquetes, paquetes premium, hipotecas) + **escala** (log_activo, préstamos sobre activo). La señal está en lo que **ofrecen**, no tanto en el balance.

- **Chicos en transformación**: dominados por **ROA negativo** (|SHAP| = 3.30) + **eficiencia mala** (1.43). Cuando esas dos pegan, no importa nada más. La oferta de productos casi no influye en esta clase.

- **Mayoristas/inversión**: definidos por **ausencia de mora** (`cartera_irregular` baja, |SHAP| = 2.39) + **eficiencia muy baja** (cost-to-income muy bajo, típico de banca corporativa) + **préstamos bajos sobre activo** (titulizan/invierten en lugar de prestar).

### 6.3 Lectura de los summary plots

- **Minoristas masivos**: SHAP positivo cuando las binarias de oferta están en 1 (rojo) y cuando ROA, log_activo y préstamos sobre activo son altos. Negativo cuando `cartera_irregular` es alta. Es la combinación "presencia de retail completo + balance sano".

- **Chicos en transformación**: la cola izquierda del ROA (azul → ROA bajo/negativo) tiene SHAP de hasta +8. Los puntos rojos de `eficiencia` (eficiencia alta = ratio gastos/ingresos malo) también empujan fuerte hacia esta clase. El resto está casi en cero — **pocas variables, muy fuertes**.

- **Mayoristas/inversión**: la cola azul de `cartera_irregular` (mora baja) tiene SHAP positivo hasta +7. Lo definen por **ausencias**: sin mora, sin paquetes, sin personales, eficiencia baja.

## 7. Interpretación para la tesis

### Lo que el clasificador valida

1. **La partición del 05A es estructura real, no artefacto del GMM**. Un modelo supervisado que nunca vio el algoritmo de clustering aprende a reproducir las etiquetas con accuracy 0.867 en CV anidado. La señal está en los datos.

2. **Las features que SHAP prioriza coinciden con la narrativa del resumen 05A**. ROA, cartera irregular, eficiencia, oferta de paquetes premium e hipotecas — las mismas variables que el resumen usaba para describir cada perfil. El clasificador "redescubre" la misma estructura interpretativa sin haberla recibido como input.

3. **Los errores son los bordes esperados**. BROU (outlier extremo), Dino y Columbia (consumer finance ambigua), Banco Municipal Rosario y Coinag (minoristas deteriorados al borde de la transformación). El resumen del 05A ya los identificaba como casos intermedios — el clasificador independientemente llega a la misma conclusión.

### Lo que el clasificador NO valida

- **NO valida que k=3 sea el k óptimo** ni que GMM-A sea mejor que GMM-B, K-Means del 03 o GMM del 04. Solo dice: dada la partición de GMM-A, es coherente con las features. Para comparar entre modelos habría que correr el mismo pipeline sobre las etiquetas del 03 y 04 y comparar.

- **NO captura la dimensión "digital"** que el notebook 04 había identificado. Los 4 digitales (Brubank, Uala, Voii, Dino) están dentro del cluster "Chicos transformación" en el 05A, y el clasificador los maneja como tales. Dino se confunde porque es atípico incluso entre digitales. Si la digitalización es una pregunta de investigación central, habría que tratarla como dimensión aparte.

### Próximos usos del modelo

- **Clasificación de nuevos cortes** (Dic-26+) sin reentrenar el GMM.
- **Análisis de transiciones**: aplicar el clasificador a los cortes Dic-23 y Dic-25 por separado y ver qué bancos migraron de perfil.
- **Posible extensión**: agregar tasas y comisiones de `oferta_banco.csv` como features continuas y ver si emergen sub-perfiles dentro del cluster minorista.

## 8. Limitaciones

- **n=52 es chico**. La estimación honesta (outer CV) tiene desvío 0.075 — el accuracy real podría estar entre 0.79 y 0.94 con 95 % de confianza aproximado. El modelo es bueno pero la precisión de "qué tan bueno" es limitada.
- **El fold 3 con accuracy 1.000** sugiere que la dificultad varía entre folds. Es un artefacto de tamaño, no de sesgo.
- **SHAP sobre el modelo final** se entrenó con todos los datos — los valores SHAP son interpretativos sobre la estructura aprendida, no son métricas de performance.
- **El modelo "aprende" las etiquetas del GMM**, así que reproduce sus sesgos. Si el GMM-A puso a Banco Dino en "chicos transformación" pero lo "correcto" desde el negocio sería "digital", el clasificador hereda el error.

## 9. Síntesis para la tesis

El clasificador LightGBM aplicado sobre las etiquetas del modelo principal (05A) cierra el análisis con resultados sustantivos:

1. **Accuracy 0.867 (macro-F1 0.850)** en CV anidado de 5×3 folds — la partición del clustering es predecible y reproducible.
2. **SHAP confirma la narrativa interpretativa del 05A**: cada cluster se define por una combinación distintiva de features (oferta retail + escala para minoristas; ROA negativo + eficiencia mala para chicos; ausencia de mora + bajo retail para mayoristas).
3. **Los errores son los bordes esperados** — los mismos casos que el resumen del 05A había marcado como intermedios.

Estos tres resultados, juntos, **transforman el silhouette bajo de 0.145 de una preocupación en información**: el sistema bancario argentino tiene perfiles con fronteras borrosas, pero esos perfiles son lo suficientemente robustos como para ser aprendidos por un clasificador independiente. La elección del 05A como modelo principal se sostiene.
