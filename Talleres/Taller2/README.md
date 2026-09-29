# Taller 2 — Ambientes virtuales de Python

Machine Learning Engineering (CC3105). Un taller corto para aprender (o repasar) cómo funcionan los ambientes virtuales de Python con `venv` + `pip` y con Conda, aplicado a proyectos de Machine Learning.

```
Talleres/Taller2/
├── requirements.txt         # paquetes de ML con versiones fijadas (punto 2)
├── requirements-lock.txt    # pip freeze completo del venv (incluye dependencias transitivas)
├── environment.yml          # equivalente para conda (punto 4, reto)
├── check_env.py             # imprime el intérprete y las versiones del ambiente activo
├── capturas/                # capturas de pantalla (entregables 2 y 3)
└── pipeline/                # pipeline de scikit-learn empaquetado (punto 3 + reto)
    ├── requirements.txt     # dependencias de ejecución, fuente única para pyproject.toml
    ├── requirements-dev.txt
    ├── MANIFEST.in
    └── pyproject.toml
```

---

## 1. Documentación: cómo crear un ambiente virtual

- **Python, tutorial:** [12. Virtual Environments and Packages](https://docs.python.org/3/tutorial/venv.html). Explica el problema que resuelven y el flujo `python -m venv` → `activate` → `pip install`.
- **Python, referencia:** [`venv` — Creation of virtual environments](https://docs.python.org/3/library/venv.html). Opciones del módulo (`--system-site-packages`, `--clear`, `--upgrade-deps`, `--prompt`) y cómo funciona (`pyvenv.cfg`, `sys.prefix` vs `sys.base_prefix`).
- **VS Code:** [Python environments in VS Code](https://code.visualstudio.com/docs/python/environments). Cubre el comando *Python: Create Environment* (crea `.venv` o un ambiente conda y ofrece instalar desde `requirements.txt`) y *Python: Select Interpreter*.

Un ambiente virtual es un directorio con su propio intérprete (un enlace al Python base) y su propio `site-packages`. Al activarlo, `python` y `pip` apuntan a ese directorio, así que los paquetes se instalan aislados del sistema y de otros proyectos.

## 2. Ambiente virtual y archivo de requisitos

Por convención el archivo se llama **`requirements.txt`**. Lo lee `pip install -r` y tiene un requisito por línea con el formato de [PEP 508](https://peps.python.org/pep-0508/) (`paquete==versión`, `>=`, `~=`, extras, marcadores). Referencia: [pip — Requirements File Format](https://pip.pypa.io/en/stable/reference/requirements-file-format/).

```bash
python -m venv .venv                 # crear
source .venv/bin/activate            # activar (Windows: .venv\Scripts\activate)
pip install -r requirements.txt      # instalar los paquetes de ML
python check_env.py                  # evidencia: el intérprete es el del .venv
pip freeze > requirements-lock.txt   # congelar TODO lo instalado (incluye transitivas)
deactivate
```

`requirements.txt` fija solo las dependencias directas (numpy, pandas, scipy, scikit-learn, joblib, matplotlib, seaborn, jupyter). `requirements-lock.txt` es la salida de `pip freeze`, con las más de 100 dependencias transitivas. Sirve para reproducir el ambiente exactamente, byte por byte.

## 3. Requisitos del pipeline de scikit-learn

`pipeline/` contiene el paquete `data_prep_pipeline` de [Actividad 1](../../Actividades/Actividad1): extracción → filtrado → split estratificado → `ColumnTransformer` sobre el dataset Telco Customer Churn. Sus dependencias de ejecución están en `pipeline/requirements.txt` y las herramientas de desarrollo en `pipeline/requirements-dev.txt`.

### Reto: incluir el archivo de requisitos en el paquete

Normalmente las dependencias se escriben dos veces: en `requirements.txt` y en `dependencies = [...]` de `pyproject.toml`. Con el tiempo las dos listas dejan de coincidir. En este caso `requirements.txt` es la única fuente y el paquete la lee al construirse:

```toml
[project]
dynamic = ["dependencies", "optional-dependencies"]

[tool.setuptools.dynamic]
dependencies = { file = ["requirements.txt"] }
optional-dependencies.dev = { file = ["requirements-dev.txt"] }
```

Para que esto funcione hizo falta una configuración adicional: **`MANIFEST.in`**. `python -m build` primero genera el sdist (`.tar.gz`) y luego construye el wheel *a partir de ese sdist*. Si los requirements no van dentro del sdist, el wheel no puede leer sus dependencias y la construcción falla.

```
include requirements.txt
include requirements-dev.txt
```

Resultado verificado:

```bash
cd pipeline && python -m build
tar tzf dist/data_prep_pipeline-0.7.0.tar.gz | grep requirements
#   data_prep_pipeline-0.7.0/requirements-dev.txt
#   data_prep_pipeline-0.7.0/requirements.txt
unzip -p dist/data_prep_pipeline-0.7.0-py3-none-any.whl '*/METADATA' | grep Requires-Dist
#   Requires-Dist: scikit-learn>=1.1
#   Requires-Dist: pandas>=1.5
#   Requires-Dist: numpy>=1.21
#   Requires-Dist: pytest; extra == "dev"
#   Requires-Dist: build; extra == "dev"
```

Para correr el pipeline:

```bash
cd pipeline
pip install -e ".[dev]"
python scripts/download_data.py
python scripts/verify_pipeline.py   # -> PIPELINE OK
pytest
```

## 4. Ambientes con Anaconda / Conda

- **Documentación:** [Managing environments — conda docs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Incluye las secciones *Creating an environment with commands*, *Creating an environment from an environment.yml file* y *Sharing an environment*.

```bash
conda create -n taller2-ml python=3.12 scikit-learn pandas   # desde la línea de comandos
conda env create -f environment.yml                          # desde el archivo
conda activate taller2-ml
python check_env.py
conda env list
conda env export --from-history > environment.yml            # exportar solo lo pedido explícitamente
conda deactivate
```

### Reto: ¿se puede definir un archivo de requisitos en conda?

Sí. Por convención se llama **`environment.yml`** y tiene algunas diferencias con `requirements.txt`:

| | `requirements.txt` (pip) | `environment.yml` (conda) |
|---|---|---|
| Formato | Texto plano, un requisito por línea | YAML |
| Versión de Python | No la controla; usa el Python con el que se creó el venv | Se fija como un paquete más (`python=3.12`) |
| Origen de paquetes | PyPI (o índices con `--index-url`) | Canales (`conda-forge`, `defaults`, …) |
| Paquetes no-Python | No (solo wheels/sdists de Python) | Sí: CUDA, MKL, compiladores, librerías de C |
| Nombre del ambiente | No aplica | `name:` dentro del archivo |
| Mezclar gestores | No | Sí, con una sección `pip:` |
| Sintaxis de versión | `==`, `>=`, `~=` | `=` (prefijo, `numpy=2.5` → 2.5.x), `==`, `>=` |
| Congelar | `pip freeze` | `conda env export` / `conda list --explicit` |

`conda create --file` también acepta un archivo plano (`conda list --export > spec.txt`), pero ese formato usa `paquete=versión=build` y no es compatible con pip.

---

## Tabla comparativa: venv + pip / Conda vs. uv

Fuente: [Python UV: The Ultimate Guide to the Fastest Python Package Manager — DataCamp](https://www.datacamp.com/tutorial/python-uv).

| Criterio | `venv` + `pip` | Conda | uv |
|---|---|---|---|
| Qué es | Módulo estándar + instalador oficial de PyPA | Gestor de paquetes y ambientes multi-lenguaje | Gestor de paquetes y proyectos escrito en Rust (Astral) |
| Instalación | Viene con Python | Instalador de Anaconda / Miniconda / Miniforge | Binario independiente (`curl … \| sh`, `pip install uv`) |
| Velocidad | Base | Lenta (resolver SAT, aunque `libmamba` mejoró) | 10–100× más rápido que pip gracias a resolución paralela y caché global con hardlinks |
| Versiones de Python | No instala Python | Sí (`python=3.12`) | Sí (`uv python install 3.12`) |
| Crear ambiente | `python -m venv .venv` | `conda create -n env` | `uv venv` (o automático con `uv sync` / `uv run`) |
| Añadir paquete | `pip install x` (+ editar requirements a mano) | `conda install x` | `uv add x` (actualiza `pyproject.toml` y `uv.lock`) |
| Archivo de requisitos | `requirements.txt` | `environment.yml` | `pyproject.toml` (PEP 621) |
| Lockfile | No nativo (`pip freeze`, pip-tools) | No nativo (`conda-lock` externo) | `uv.lock`, universal y multiplataforma |
| Compatibilidad pip | — | Sección `pip:` | `uv pip install -r requirements.txt` como reemplazo directo |
| Paquetes no-Python (CUDA, MKL) | No | **Sí**, su principal ventaja | No, solo PyPI |
| Ejecutar sin activar | `.venv/bin/python` | `conda run -n env` | `uv run script.py` |
| Herramientas CLI globales | pipx (aparte) | — | `uvx` / `uv tool install` |
| Construir y publicar | `build` + `twine` | `conda-build` (para canales conda) | `uv build`, `uv publish` |
| Ubicación del ambiente | En el proyecto (`.venv`) | Centralizada (`~/miniforge3/envs/`) | En el proyecto (`.venv`) |
| Cuándo usarlo | Proyectos simples, sin dependencias extra | Stacks científicos o de GPU con librerías nativas | Proyectos Python modernos, CI rápido, reproducibilidad |

## Tabla comparativa: venv + pip / Conda vs. Poetry

Fuente: [Python Poetry: Modern And Efficient Python Environment And Dependency Management — DataCamp](https://www.datacamp.com/tutorial/python-poetry).

| Criterio | `venv` + `pip` | Conda | Poetry |
|---|---|---|---|
| Qué es | Módulo estándar + instalador | Gestor de paquetes y ambientes multi-lenguaje | Gestor de dependencias y empaquetado escrito en Python |
| Instalación | Viene con Python | Instalador de Anaconda / Miniforge | `pipx install poetry` o el instalador oficial |
| Iniciar proyecto | Manual | Manual | `poetry new` / `poetry init` (genera estructura y `pyproject.toml`) |
| Archivo de requisitos | `requirements.txt` | `environment.yml` | `pyproject.toml` (`[project]` desde Poetry 2.0, antes `[tool.poetry]`) |
| Lockfile | No nativo | No nativo | `poetry.lock`, con hashes y resolución determinista |
| Resolución de dependencias | Backtracking de pip, sin garantía global | SAT (conda / libmamba) | Resolver propio que detecta conflictos antes de instalar |
| Añadir paquete | `pip install x` + editar a mano | `conda install x` | `poetry add x` (actualiza `pyproject.toml` y lock) |
| Grupos de dependencias | Archivos separados (`requirements-dev.txt`) | Archivos separados | `poetry add --group dev pytest` |
| Manejo del ambiente | Manual (`venv` + `activate`) | `conda activate` | Automático: `poetry install` crea el venv, `poetry run` / `poetry env activate` |
| Versiones de Python | No | Sí | No las instala; usa las del sistema o pyenv (`poetry env use 3.12`) |
| Paquetes no-Python | No | **Sí** | No |
| Construir y publicar | `build` + `twine` | `conda-build` | `poetry build`, `poetry publish` integrados |
| Exportar a requirements | — | — | `poetry export -f requirements.txt` (plugin) |
| Velocidad | Base | Lenta | Similar a pip o más lenta al resolver; mucho más lenta que uv |
| Cuándo usarlo | Scripts y proyectos simples | Stacks científicos o de GPU | Librerías que se publican en PyPI y equipos que necesitan builds deterministas |

---

## Capturas de pantalla

Las capturas están en [`capturas/`](capturas/).

1. **venv:** `python -m venv .venv`, `source .venv/bin/activate`, `pip install -r requirements.txt`, `python check_env.py` (el ejecutable apunta a `.venv/bin/python`).
2. **Conda:** `conda env create -f environment.yml`, `conda env list`, `conda activate taller2-ml`, `python check_env.py` (el ejecutable apunta a `…/envs/taller2-ml/bin/python`).
3. **Pipeline:** `python scripts/verify_pipeline.py` → `PIPELINE OK`.

## Conclusiones

1. **Reproducibilidad de experimentos.** Un modelo depende tanto del código como de las versiones exactas de numpy, scikit-learn, etc. Un cambio menor (por ejemplo, un default distinto en `OneHotEncoder` o en un solver) cambia métricas o rompe la carga de un modelo guardado con `joblib`/`pickle`. Fijar el ambiente (`requirements-lock.txt`, `uv.lock`, `poetry.lock`) permite volver a entrenar y obtener los mismos resultados meses después.
2. **Un ambiente por proyecto o por etapa.** En un mismo equipo pueden convivir proyectos con TensorFlow y PyTorch, o con versiones de CUDA incompatibles. Incluso dentro de un proyecto conviene separar el ambiente de exploración (Jupyter, matplotlib, seaborn), el de entrenamiento y uno mínimo de *serving*, que solo necesita scikit-learn y pandas y produce imágenes Docker más pequeñas y seguras.
3. **Paridad entre desarrollo, CI y producción.** El mismo archivo de requisitos se usa en la laptop, en GitHub Actions, en el `Dockerfile` y en Databricks. Esto elimina el clásico «en mi máquina sí funciona». El reto del punto 3 muestra una forma de mantener una sola fuente de verdad: el paquete lee sus dependencias del mismo `requirements.txt` que usa el equipo.
4. **Elegir la herramienta según el tipo de dependencias.** `venv` + `pip` basta para la mayoría de proyectos de scikit-learn. Conda gana cuando hacen falta librerías nativas (CUDA, cuDNN, MKL, GDAL) porque las instala como paquetes. uv es la opción más rápida y moderna para proyectos puramente Python, con lockfile y manejo de versiones de Python incluidos. Poetry sigue siendo sólido para publicar librerías.
5. **Seguridad y auditoría.** Un ambiente declarado en un archivo se puede auditar (`pip-audit`, Dependabot), actualizar de forma controlada y revisar en un PR. En cambio, instalar paquetes a mano en el Python global no deja rastro y mezcla dependencias de proyectos distintos.
