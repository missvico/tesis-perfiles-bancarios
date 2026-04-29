# Resumen de hallazgos — Notebook 02 (EDA)

**Panel:** 168 observaciones = 56 bancos × 3 cortes (Dic-23 / Dic-24 / Dic-25), con ~13 ratios construidos sobre balance + variables de oferta comercial del régimen de transparencia.

## 1. Estructura del sistema

- **Fuerte asimetría de tamaño:** el activo va de ~7 M a ~48.000 M (máx / mín ≈ 7.000×). Justifica trabajar en escala log y con mediana en vez de media, y usar **Spearman** en correlaciones.
- Composición estable en los tres cortes: 30 privados nacionales, 17 públicos, 9 extranjeros.
- **Concentración moderada y estable**: HHI sobre activo pasa de 1.005 (Dic-23) a 1.056 (Dic-25); el top-5 explica ~60 % del activo en los tres cortes. No hubo un shock estructural de concentración en el período.

## 2. Diferencias estructurales por tipo de entidad (Dic-2024)

| Dimensión | Públicos | Privados nac. | Extranjeros |
|---|---|---|---|
| Activo mediano | **1.085 M** (mayor) | 216 M | 476 M |
| ROA / ROE | Altos (2,99 / 13,93) | Medios (2,21 / 12,90) | Bajos (0,17 / 0,60) |
| Apalancamiento | 3,67 | **4,51** (más apalancados) | 2,50 (más conservadores) |
| Depósitos/Activo | 63,7 % | 65,0 % | **43,4 %** (menos fondeo minorista) |
| Cartera irregular | **2,17 %** | 0,58 % | 1,11 % (sobre 5/9 aplicables) |
| Liquidez | 13,4 % | 21,4 % | **33,1 %** |
| Eficiencia (gastos/margen) | 33,8 % (mejor) | 48,0 % | 35,5 % |
| Dotación mediana | **750 empleados** | 192 | 173 |
| Activo/empleado | 1,42 M | 1,36 M | **2,23 M** (más productivo) |
| Productos ofrecidos | 5 | 3,5 | 1 |

**Lectura:** los tres tipos tienen **modelos de negocio distintos**, no es solo una cuestión de escala. Públicos = grandes, alta cobertura, más riesgo en cartera, buena eficiencia operativa por escala. Privados nacionales = intermedios, más apalancados, fondeados con depósitos, peor eficiencia (margen comprimido). Extranjeros = chicos, líquidos, productivos, oferta acotada, modelo mayorista.

### 2.1 Respaldo estadístico — Kruskal-Wallis (Dic-2024)

Contraste no paramétrico de igualdad de distribuciones entre los 3 tipos (n=17/30/9). H₀: vienen de la misma población.

| Variable | H | p-valor | Significativo 5 % |
|---|---:|---:|:---:|
| Liquidez | 13,33 | 0,0013 | ✅ |
| Apalancamiento | 9,10 | 0,0106 | ✅ |
| ROA | 6,85 | 0,0325 | ✅ |
| Cartera irregular | 6,67 | 0,0356 | ✅ |
| N° productos ofrecidos | 6,43 | 0,0401 | ✅ |
| ROE | 6,20 | 0,0451 | ✅ |
| Depósitos / Activo | 3,95 | 0,1386 | ❌ |
| Activo / empleado | 1,65 | 0,4391 | ❌ |

**6 de 8 variables distinguen los tres perfiles al 5 %.** Las dos que no discriminan (depósitos/activo y activo/empleado) reflejan que los extranjeros son heterogéneos entre sí (mayoristas vs. retail) y los dos grupos nacionales no se diferencian fuerte en esas dimensiones. La afirmación "los tres perfiles son estadísticamente distinguibles" queda respaldada formalmente a pesar del n=9 en extranjeros.

## 3. Evolución Dic-23 → Dic-25

- **Caída generalizada de rentabilidad**: ROA mediano en públicos pasa de 4,15 → 0,79; privados nac. de 3,06 → 0,35; extranjeros oscilan (4,10 → 0,17 → 2,44). Se verificó que el corte Dic-2025 es anual, no parcial — la caída es real, no un artefacto de no-anualización.
- **Deterioro de cartera**: irregular en públicos trepa de 2,33 % a **5,05 %**; privados nac. de 1,85 % a 3,38 %; extranjeros se mantienen bajos pero sobre una muestra pequeña (sólo 5 de los 9 tienen cartera minorista relevante).
- **Recomposición del balance hacia préstamos**: préstamos/activo sube en los tres grupos (~14 % → ~38 % en privados nacionales). Consistente con la reactivación del crédito post-2023.
- **Salto de productividad**: activo/empleado ~duplica en todos los tipos — reflejo de ajuste de dotación y/o de precios.
- **Eficiencia se deteriora** en los tres tipos entre 2023 y 2025 (mayor consumo de margen por gastos de administración), aunque los públicos mantienen ventaja relativa por escala.

## 4. Correlaciones destacables

- **Tamaño ↔ rentabilidad**: relación débil; ROA no depende fuertemente del activo.
- **Cartera irregular ↔ ROA**: correlación negativa esperada.
- **Apalancamiento ↔ ROE**: positiva — los más apalancados rentan más sobre patrimonio.
- **Depósitos/Activo ↔ Préstamos/Activo**: el perfil de fondeo predice el perfil de colocación.

## 5. Oferta comercial (transparencia BCRA)

- Cobertura desigual de productos: hipotecas en 25 bancos, prendarios sólo 19, plazo fijo en 50.
- **Variedad por tipo (mediana):** públicos lideran en hipotecas (2), tarjetas (6,5) y paquetes (3,5); privados nacionales concentran plazo fijo (11); extranjeros lideran en prendarios.
- **Premium/gold/clásica**: el mix de segmentación difiere por tipo — los privados y extranjeros concentran más oferta premium.
- **Balance ↔ oferta**: la correlación tamaño-TEA y ROA-TEA es baja; las diferencias de tasas parecen explicarse más por estrategia comercial que por rentabilidad o escala.

## 6. Notas metodológicas

- **Signo de la eficiencia (corregido)**: el BCRA publica egresos financieros, egresos por servicios y gastos de administración con signo negativo en las tablas HTML. La primera corrida del EDA daba eficiencia negativa (-13 %/-17 %/-20 %) por tomar los valores directos; la versión final aplica valor absoluto sobre esas columnas antes de calcular el margen y el ratio, devolviéndolo al rango estándar ~30-80 %.
- **Cartera irregular de bancos sin cartera minorista**: 4 de los 9 extranjeros (Bank of China, JPMorgan, BNP Paribas, Cetelem) no reportan financiaciones relevantes. Con `fillna(0)` quedaban mal clasificados como "0 % de mora"; ahora se les asigna `NaN` y las medianas por tipo se calculan sólo sobre los bancos donde la métrica aplica.
- **Missings por banco**: sobre el set de 15 variables candidatas al clustering, Cetelem concentra 53 % de missings, y BNP / Bank of China tienen 40 %. Son candidatos a excluirse (o tratarse aparte) en el notebook 03.

## 7. Conclusiones para la tesis

1. **Existen al menos tres perfiles claramente distinguibles** por tipo de entidad — la hipótesis de heterogeneidad estructural queda soportada tanto descriptivamente como por Kruskal-Wallis en 6 de 8 variables clave.
2. Hay **suficiente dispersión dentro de cada tipo** (sobre todo privados nacionales) para que un clustering no-supervisado encuentre subgrupos interesantes.
3. El **período 2023-2025 no es estático**: hay tendencia de deterioro de rentabilidad y cartera + recomposición del balance hacia préstamos que conviene reconocer al segmentar (¿clustering por corte o sobre promedios?).
4. La **oferta comercial agrega información independiente** al balance — vale incluirla en el vector de features.
5. El **clustering debe excluir (o tratar separadamente) los bancos mayoristas / de inversión** cuyo modelo de negocio hace que buena parte de las métricas de balance retail no apliquen.

---

**Artefactos asociados:**
- `data/processed/panel_ratios.csv` (168 × 157) — panel transversal con ratios corregidos
- `data/processed/oferta_banco.csv` (51 × 5) — tasas/comisiones/requisitos agregados por banco
- `data/processed/variedad_productos.csv` (51 × 7) — conteo de productos únicos por banco y categoría
