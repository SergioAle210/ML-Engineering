# data-prep-pipeline (Taller 2)

Pipeline de scikit-learn (extracción → filtrado → split → preprocesamiento) para el dataset
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn),
tomado de [`Actividades/Actividad1`](../../../Actividades/Actividad1) y reempaquetado para el reto del Taller 2:
las dependencias del paquete se declaran desde `requirements.txt`.

## Archivos de requisitos

| Archivo | Uso |
|---|---|
| `requirements.txt` | Dependencias de ejecución. `pyproject.toml` las lee con `[tool.setuptools.dynamic]`. |
| `requirements-dev.txt` | pytest y build. Se exponen como el extra `[dev]`. |
| `MANIFEST.in` | Incluye ambos archivos en el sdist; sin él, construir el wheel desde el sdist falla. |

## Uso

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt   # o: pip install -e ".[dev]"
pip install -e .
python scripts/download_data.py
python scripts/verify_pipeline.py
pytest
python -m build        # genera dist/*.tar.gz y dist/*.whl
```
