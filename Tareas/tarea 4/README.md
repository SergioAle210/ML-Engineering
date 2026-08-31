# example_package_chuyr3

Paquete de ejemplo creado siguiendo el tutorial oficial de empaquetado de Python.
Guia: https://packaging.python.org/en/latest/tutorials/packaging-projects/

Se publico en Test PyPI con fines educativos, como parte de la tarea 4 de MLOps.

## Contenido del proyecto

1. src/example_package_chuyr3: codigo fuente del paquete, con la funcion add_one.
2. pyproject.toml: configuracion del paquete en formato TOML, siguiendo el estandar actual de empaquetado en Python.
3. demo_uso.py: script que instala el paquete desde Test PyPI en un entorno limpio y prueba su funcionamiento.
4. LICENSE: licencia MIT del proyecto.

## Como construir el paquete

Dentro de un entorno virtual con las herramientas build y twine instaladas, ejecutar:

    python -m build

Esto genera los archivos .whl y .tar.gz dentro de la carpeta dist.

## Como se publico

El paquete se subio a Test PyPI usando twine, autenticando con un token de API generado en la cuenta de Test PyPI:

    python -m twine upload --repository testpypi dist/*

Version publicada: 0.0.1
Pagina del proyecto: https://test.pypi.org/project/example-package-chuyr3/

## Como probar el paquete publicado

El script demo_uso.py instala el paquete desde Test PyPI en un entorno virtual limpio y llama a la funcion add_one con varios valores. Pasos:

    python -m venv .venv-test
    .venv-test\Scripts\Activate.ps1
    pip install --index-url https://test.pypi.org/simple/ --no-deps example_package_chuyr3
    python demo_uso.py
