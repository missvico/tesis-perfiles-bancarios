# Checklist del repositorio para la Entrega Final

Estado del repo relevado el 2026-07-04. Objetivo: que el repo público sirva como anexo
de código reproducible del Trabajo Final (la consigna exige: código comentado, instrucciones
de reproducibilidad y documento con versiones de software).

---

## ✅ PRESERVAR (ya trackeado — no tocar)

### Código (el corazón de la entrega)
| Archivo | Rol |
|---|---|
| `scripts/00_scraper_bcra.py` | Captura de datos del portal BCRA (etapa 1 del pipeline) |
| `notebooks/01_ingesta_y_calidad.ipynb` | Integración, ratios, diagnóstico de calidad |
| `notebooks/02_eda.ipynb` | EDA completo (§4.3 del paper) |
| `notebooks/03_clustering.ipynb` | K-Means base + Ward + DBSCAN (modelos de contraste) |
| `notebooks/04_clustering_alternativo.ipynb` | GMM solo-continuas (modelo de contraste) |
| `notebooks/05_clustering_intermedio.ipynb` | **Modelo principal** (GMM espacio mixto) |
| `notebooks/06_clasificador_lgbm.ipynb` | Validación supervisada + SHAP |
| `requirements.txt` | Dependencias (ver pendiente de versiones abajo) |

### Datos (necesarios para reproducibilidad — total ~18 MB, tamaño OK para GitHub)
- `data/raw/bcra_scraping/**` — CSVs crudos del scraper. **Imprescindibles**: el portal del
  BCRA cambia con el tiempo; sin estos archivos el análisis no es reproducible tal cual se corrió.
- `data/raw/transparencia/*.CSV` — Régimen de Transparencia (snapshot). Ídem: irrecuperable a futuro.
- `data/processed/**` — outputs intermedios (panel_ratios.csv, oferta_banco.csv, etc.).
  Permiten reproducir desde el notebook 02 en adelante sin re-scrapear.
- Son datos **públicos** del BCRA: no hay problema legal ni de privacidad en publicarlos.

### Documentación del análisis (el paper los referencia en el Anexo 8.2)
- `analysis/resumen_eda.md`
- `analysis/resumen_clustering.md`
- `analysis/resumen_clustering_alternativo.md`
- `analysis/resumen_clasificador.md`
- `analysis/figs_entrega3/` (figuras del paper) y `analysis/figs_05/` (referenciadas por los .md)
- `analysis/entrega3.md` (documenta el proceso; inofensivo y útil como historia)
- `SPEC.md` (especificación del proyecto)
- `.gitignore`

### Pendiente de commit (nuevos, van al repo)
- `analysis/figs_entrega3/fig_eda_spearman.png`
- `analysis/figs_entrega3/fig_eda_boxplots_tipo.png`
- `analysis/figs_entrega3/fig_eda_oferta.png`
- `analysis/entrega_final.tex` (opcional pero recomendado: es la fuente del PDF entregado)

---

## ❌ SACAR del repo (o no commitear nunca)

### Scripts auxiliares que ya cumplieron su fin (hoy trackeados — candidatos a `git rm`)
| Archivo | Por qué sacarlo |
|---|---|
| `scripts/insert_section7.py` | Patch one-shot ya aplicado; referencia estados intermedios que ya no existen |
| `scripts/patch_notebook_02.py` | Ídem |
| `scripts/patch_notebook_03.py` | Ídem |

**Razón**: un evaluador que los corra hoy rompería los notebooks o no haría nada. Confunden
sobre cuál es la "fuente de verdad".

### Discutible — decisión tuya
| Archivo | Consideración |
|---|---|
| `scripts/gen_notebook_03/04/05/06.py` | Son los generadores de los notebooks. **A favor de dejarlos**: documentan la génesis. **En contra**: duplican el contenido y un evaluador puede dudar de si debe correr el .py o el .ipynb. **Recomendación**: sacarlos del repo (o moverlos a `scripts/dev/` con una línea en el README aclarando que son herramientas de desarrollo, no parte del pipeline). |

### Nunca commitear (ya cubierto por .gitignore o sin trackear — verificado)
- `venv/` — entorno virtual (ya ignorado)
- `.claude/` — configuración local del asistente (ya ignorado; **nunca** debe ir a un repo público)
- `notebooks/*.ipynb.bak` — backups (ya ignorados)
- `.DS_Store` — (ya ignorado)
- `analysis/figs_03/`, `figs_04/`, `figs_06/` — extracciones de trabajo, sin referencias desde
  ningún documento versionado. Dejar locales, no commitear.
- `claude-memory-backup/`, `*.credentials.json` — ya ignorados (no existen hoy, pero el
  .gitignore los cubre por si acaso)

**Verificado**: no hay credenciales, tokens ni API keys en ningún archivo trackeado
(el scraper usa endpoints públicos sin autenticación).

---

## ⚠️ FALTA CREAR antes de entregar (exigencia explícita de la consigna)

1. **`README.md`** en la raíz, con:
   - Descripción de una línea del proyecto + link al PDF de la entrega.
   - **Instrucciones de reproducibilidad**: crear venv → `pip install -r requirements.txt` →
     (macOS: `brew install libomp` para LightGBM) → orden de ejecución:
     `00_scraper` (opcional, los datos ya están) → notebooks `01 → 02 → 03 → 04 → 05 → 06`.
   - **Versiones de software**: Python 3.14 + salida de `pip freeze` (o tabla de versiones).
     La consigna pide "un documento que indique la versión del software utilizado" —
     el `requirements.txt` actual usa `>=`, que NO fija versiones. Opciones: (a) agregar
     `requirements-freeze.txt` con `pip freeze > requirements-freeze.txt`, o (b) tabla en el README.
2. (Opcional) `LICENSE` — si el repo es público, una licencia (MIT) evita ambigüedad.

---

## Comandos sugeridos (cuando decidas ejecutarlos)

```bash
# sacar los scripts one-shot del repo (los conserva en disco)
git rm --cached scripts/insert_section7.py scripts/patch_notebook_02.py scripts/patch_notebook_03.py
echo "scripts/insert_section7.py" >> .gitignore   # opcional, para que no reaparezcan

# commitear lo nuevo
git add analysis/entrega_final.tex analysis/figs_entrega3/fig_eda_*.png analysis/checklist_repo_entrega_final.md

# generar el freeze de versiones (con el venv activado)
pip freeze > requirements-freeze.txt
```
