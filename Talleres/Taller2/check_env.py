"""Muestra qué intérprete y qué versiones de paquetes está usando el ambiente activo.

Útil para las capturas: confirma que `python` apunta al ambiente (venv o conda)
y no al Python del sistema.
"""
import platform
import sys
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib", "seaborn"]

if sys.prefix != sys.base_prefix:
    env_type = "venv (python -m venv)"
elif (Path(sys.prefix) / "conda-meta").is_dir():
    env_type = "conda"
else:
    env_type = "Python del sistema (ningún ambiente activo)"
print(f"python:      {platform.python_version()}")
print(f"ejecutable:  {sys.executable}")
print(f"sys.prefix:  {sys.prefix}")
print(f"tipo:        {env_type}")
print()
for name in PACKAGES:
    try:
        print(f"{name:<14}{version(name)}")
    except PackageNotFoundError:
        print(f"{name:<14}(no instalado)")
