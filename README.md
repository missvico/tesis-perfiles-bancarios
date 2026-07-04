# Perfiles bancarios en Argentina

Una segmentación no supervisada del sistema financiero argentino (2023–2025).

Trabajo Final del Taller de Tesis I — Maestría en Explotación de Datos y Descubrimiento
del Conocimiento, FCEN-UBA. **Autora:** Victoria Di Liscia.

El trabajo identifica tres perfiles de negocio en el sistema bancario argentino
—minoristas masivos, chicos en transformación y mayoristas/inversión— mediante clustering
(GMM sobre un espacio mixto de balance + oferta comercial), validados con un clasificador
supervisado (LightGBM, accuracy 0,867 en validación cruzada anidada) e interpretados con SHAP.

## Estructura del repositorio

```
scripts/00_scraper_bcra.py      Captura de datos del portal público del BCRA
notebooks/
  01_ingesta_y_calidad.ipynb    Integración de fuentes, ratios, diagnóstico de calidad
  02_eda.ipynb                  Análisis exploratorio (EDA)
  03_clustering.ipynb           K-Means base + Ward + DBSCAN (modelos de contraste)
  04_clustering_alternativo.ipynb  GMM sobre continuas (modelo de contraste)
  05_clustering_intermedio.ipynb   ** Modelo principal: GMM sobre espacio mixto **
  06_clasificador_lgbm.ipynb    Validación supervisada (LightGBM + nested CV) + SHAP
data/raw/                       Datos crudos (scraping BCRA + Régimen de Transparencia)
data/processed/                 Datos intermedios (panel de ratios, oferta por banco, etc.)
analysis/                       Resúmenes en prosa por etapa + figuras del informe
```

Los scripts `scripts/gen_notebook_*.py` son herramientas de desarrollo que generaron los
notebooks; **no** forman parte del pipeline de análisis. La fuente de verdad son los notebooks.

## Reproducibilidad

### Requisitos

- **Python 3.14** (desarrollado con 3.14.4)
- En macOS, LightGBM requiere el runtime de OpenMP: `brew install libomp`

### Instalación

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt        # rangos mínimos
# o bien, para reproducir el entorno exacto:
pip install -r requirements-freeze.txt # versiones exactas usadas en el trabajo
```

### Orden de ejecución

Los datos crudos ya están incluidos en `data/raw/`, de modo que **no hace falta re-scrapear**
(el portal del BCRA cambia con el tiempo; los datos incluidos son el snapshot usado en el trabajo).

```bash
jupyter lab
# Ejecutar en orden: notebooks 01 → 02 → 03 → 04 → 05 → 06
```

- `01` reconstruye `data/processed/` a partir de `data/raw/`.
- `02` genera el panel de ratios y los agregados de oferta comercial.
- `03`–`05` corren los modelos de clustering (el `05` contiene el modelo principal).
- `06` corre la validación supervisada (nested CV, ~2-3 minutos) y el análisis SHAP.

Todas las semillas aleatorias están fijadas (`random_state`) — los resultados son deterministas.

Para re-capturar datos desde cero (opcional): `python scripts/00_scraper_bcra.py`.

### Versiones de software

Entorno completo en [`requirements-freeze.txt`](requirements-freeze.txt). Principales:

| Paquete | Versión |
|---|---|
| Python | 3.14.4 |
| pandas | 3.0.2 |
| numpy | 2.4.4 |
| scikit-learn | 1.8.0 |
| lightgbm | 4.6.0 |
| shap | 0.52.0 |
| scipy | 1.17.1 |
| matplotlib / seaborn | 3.10.8 / 0.13.2 |

## Fuentes de datos

Todos los datos provienen de fuentes **públicas** del BCRA:

- Portal de Entidades Financieras (estados contables, situación de deudores, estructura).
- Régimen de Transparencia, Sección 36 (información comercial de productos).
