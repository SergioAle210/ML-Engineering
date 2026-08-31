"""
Demo de uso del paquete example_package_chuyr3 instalado desde Test PyPI.

Antes de correr este script, instala el paquete desde Test PyPI en un entorno limpio:

    python -m venv .venv-test
    .venv-test\\Scripts\\Activate.ps1
    pip install --index-url https://test.pypi.org/simple/ --no-deps example_package_chuyr3

Luego ejecuta:

    python demo_uso.py
"""

from importlib.metadata import version
from example_package_chuyr3 import example

print("Paquete instalado:", "example_package_chuyr3")
print("Version instalada:", version("example_package_chuyr3"))
print()

for n in [0, 5, 41, -1]:
    print(f"add_one({n}) = {example.add_one(n)}")
