# POC de puntos dinámicos de MiMcDonald's

Desarrollamos una prueba de concepto local para analizar un sistema de puntos dinámicos. Organizamos el trabajo en cinco etapas: generación sintética → limpieza → tablas de negocio → segmentación → reglas, movimientos e insights.

Comparamos una política fija con una dinámica sobre las mismas compras sintéticas. Estudiamos la distribución de los incentivos y su emisión adicional de puntos; la medición de ventas incrementales requiere un piloto con datos reales.

## Preparación y ejecución

Desde `Casos-Estudio/Caso-Estudio-2/`, en Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter lab
```

En Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Ejecutar todas las celdas de cada notebook en orden **01 → 02 → 03 → 04 → 05**, con el kernel del entorno preparado. Los notebooks pueden iniciarse desde la carpeta del caso o desde `notebooks/`. Si cambia una etapa, volver a ejecutar todas las posteriores. En particular, 03 reemplaza `perfil_clientes` sin segmento: después se requieren 04 y 05.

| Notebook | Salidas principales |
|---|---|
| [01 — Generación](notebooks/01_generar_datos.ipynb) | Cinco CSV en Bronze |
| [02 — Limpieza](notebooks/02_bronze_a_silver.ipynb) | Cinco Parquet en Silver y cinco CSV de cuarentena |
| [03 — Gold](notebooks/03_silver_a_gold.ipynb) | `perfil_clientes`, `ventas_productos` |
| [04 — Segmentación](notebooks/04_segmentacion_ml.ipynb) | Perfil con `segmento` y `afinidad_segmento_categoria` |
| [05 — Puntos e insights](notebooks/05_puntos_e_insights.ipynb) | `reglas_puntos`, `movimientos_puntos`, comparación y gráficas |

`requirements.txt` instala además el paquete local `segmentacion-clientes` (`-e .`), que contiene el pipeline de scikit-learn y sus entry points de consola (ver [Entry points](#entry-points-de-la-segmentación)).

## Estructura

```text
data/
  bronze/       # Cinco CSV sintéticos, incluidos errores intencionales
  silver/       # Cinco tablas limpias en Parquet
  gold/         # Cinco tablas de negocio en Parquet al terminar 05
  cuarentena/   # Cinco CSV de rechazos con valores originales y motivos
notebooks/
  01_generar_datos.ipynb
  02_bronze_a_silver.ipynb
  03_silver_a_gold.ipynb
  04_segmentacion_ml.ipynb
  05_puntos_e_insights.ipynb
src/
  segmentacion_clientes/
    pipeline.py   # sklearn Pipeline (preprocesamiento + K-means), nombres, afinidad y exportación
    cli.py        # Funciones de los entry points de consola
models/         # Modelo entrenado por segmentacion-entrenar (ignorado por Git)
pyproject.toml  # Paquete segmentacion-clientes y declaración de [project.scripts]
requirements.txt
tests/
  test_silver.py
  test_segmentacion.py
```

## Datos generados

- Semilla `22434`; período fijo del 1 de enero al 31 de marzo de 2025. Enero y febrero son la ventana de análisis; marzo es la ventana de aplicación. Las fechas y horas representan la hora local de Guatemala.
- Base válida de 1,000 clientes ficticios, 28 productos (cuatro por categoría), 10 restaurantes y 20,000 compras, con entre una y cuatro líneas por compra.
- Cuatro perfiles ocultos de generación: Leales, Madrugadores, En riesgo y Ocasionales. Sus etiquetas solo existen en memoria durante la generación y no se exportan como variables para ML.
- Cuatro productos con baja probabilidad de compra y preferencias por categoría diferenciadas por perfil. Los precios y el catálogo son ilustrativos, no precios oficiales.
- Montos calculados en centavos enteros. Cada línea monetaria acumula `ceil(subtotal × 10)` puntos; cada canje cuesta `precio_en_puntos × cantidad` y tiene precio monetario cero. El saldo se revisa antes del canje, sin usar puntos ganados en esa misma compra.
- Se agregan al final 16 filas duplicadas (2 clientes, 2 productos, 2 restaurantes, 5 compras y 5 líneas), una compra y línea con monto negativo, y una compra y línea con producto inexistente. Los dos últimos casos usan identificadores nuevos; así los originales válidos quedan disponibles para la limpieza. Los conteos de los CSV superan por ello los de la base válida.

Validamos la base antes de introducir errores y releemos los CSV para comprobar las anomalías, los perfiles y el comportamiento de los productos rezagados. La misma configuración reproduce los archivos de Bronze. Los cambios en esta capa requieren volver a ejecutar las etapas posteriores.

## Limpieza Bronze → Silver

Convertimos tipos, eliminamos duplicados y validamos campos obligatorios, dominios, claves, referencias, fechas e importes con SQL en DuckDB. Los importes se guardan como `DECIMAL(12,2)`; los valores con precisión excesiva se rechazan sin redondearlos. Los duplicados exactos se comparan después de normalizar espacios y tipos; se conserva el primero. Si un ID tiene contenidos distintos, todas sus versiones van a cuarentena. Los correos compartidos entre clientes distintos también se rechazan.

Se conserva la integridad de cada factura: una línea inválida (excepto una copia exacta), un detalle ausente o repetido, o un total que no coincide con la suma de las líneas provoca el rechazo de la compra completa. Los canjes tienen importe cero; una compra formada solo por canjes puede tener total cero. Se valida el precio efectivamente cobrado sin exigir que coincida con el precio actual del catálogo.

Cada CSV de cuarentena incluye las columnas originales, `archivo_origen`, `fila_csv` y `motivos`. La fila indica el orden del registro CSV contando el encabezado; no necesariamente la línea física si un campo contiene saltos de línea. Una fila puede tener varios motivos, pero se cuenta una sola vez en el resumen.

| Tabla | Bronze | Silver | Cuarentena |
|---|---:|---:|---:|
| clientes | 1,002 | 1,000 | 2 |
| productos | 30 | 28 | 2 |
| restaurantes | 12 | 10 | 2 |
| compras | 20,007 | 20,000 | 7 |
| lineas_compra | 49,532 | 49,525 | 7 |

Los 20 rechazos son los 16 duplicados y las dos compras inválidas con sus líneas. Verificamos `Bronze = Silver + cuarentena`, las relaciones, los totales y la conservación de tipos y valores en Parquet. Las tablas limpias alimentan los perfiles de clientes, la segmentación y el cálculo de puntos.

Las pruebas adicionales usan datos pequeños en carpetas temporales dentro de este proyecto, que se eliminan al terminar:

```bash
python -m unittest discover -s tests -v
```

## Segmentación ML

Entrenamos K-means con 4 grupos sobre `perfil_clientes`, con semilla `22434` y 20 inicializaciones. Utilizamos recencia, frecuencia, gasto promedio y franja preferida. La franja se convierte en una columna por franja; todas las columnas se estandarizan y el bloque de franja se pondera por `1/√4` para que cuente como una sola variable y no domine la distancia. `categoria_preferida` solo se usa para interpretar los grupos.

Interpretamos los grupos con reglas explícitas: menor frecuencia → Ocasionales; mayor proporción de mañana → Madrugadores; mayor recencia entre los restantes → En riesgo; el último → Leales. Evaluamos la silueta para k = 2…8 y la estabilidad frente a otras semillas mediante ARI. Guardamos en Gold los perfiles con su segmento y las 28 combinaciones de afinidad entre cuatro segmentos y siete categorías.

La ejecución de referencia obtiene 250 Leales, 249 Madrugadores, 255 En riesgo y 246 Ocasionales; silueta 0.4280 y ARI mínimo 1.0000 en diez semillas alternativas. k = 6 obtiene una silueta mayor (0.4499); se mantienen cuatro grupos por la interpretación de negocio. Se exportan tablas; el notebook mantiene el modelo en memoria y el entry point `segmentacion-entrenar` lo guarda además en `models/`.

### Entry points de la segmentación

El preprocesamiento y K-means forman un único `sklearn.pipeline.Pipeline` (`construir_pipeline` en `src/segmentacion_clientes/pipeline.py`). El notebook 04 importa ese mismo código, y `pyproject.toml` lo expone como comandos de consola mediante `[project.scripts]`:

```toml
[project.scripts]
segmentacion-diagnostico = "segmentacion_clientes.cli:diagnostico"
segmentacion-entrenar = "segmentacion_clientes.cli:entrenar"
segmentacion-predecir = "segmentacion_clientes.cli:predecir"
```

Al instalar el paquete (`pip install -e .`, incluido en `requirements.txt`), pip crea estos ejecutables en el entorno. Desde la carpeta del caso, después de ejecutar 01–03:

```bash
segmentacion-diagnostico                            # inercia y silueta para k = 2…8
segmentacion-entrenar --metricas metricas.json      # equivale al notebook 04: Gold + models/segmentacion_kmeans.joblib
segmentacion-predecir --salida segmentos.csv        # aplica el modelo guardado a perfil_clientes (o a --entrada)
```

Todos aceptan `--datos` para indicar otra carpeta `data/`, y `--help` describe sus opciones. `segmentacion-entrenar` produce los mismos Parquet que el notebook 04, por lo que después se debe ejecutar 05. `tests/test_segmentacion.py` comprueba sobre una copia temporal de los datos que los entry points estén registrados, que reproduzcan Gold y que la predicción con el modelo guardado coincida con los segmentos.

## Puntos e insights

Cruzamos productos rezagados con afinidades > 1.2 para asignar ×2. Si una categoría no tiene segmentos afines, dirigimos la regla de su producto rezagado a todos los clientes. Asignamos además ×2 en hamburguesas y pollo a Madrugadores, ×1.5 de cliente a En riesgo y ×1.25 a Ocasionales. Leales recibe únicamente los incentivos de producto que corresponden a su perfil.

Se utiliza la mayor regla de producto (ID menor en empates), como máximo una regla de cliente y un tope combinado ×3. Los puntos finales se calculan con `ceil(subtotal × 10 × multiplicador)`, sin redondeo intermedio y usando `Decimal`. Enero–febrero conserva la política fija. Los canjes no acumulan y se debitan antes de acreditar los puntos de la misma compra.

| Resultado | Valor con la semilla 22434 |
|---|---:|
| Reglas generadas | 8 |
| Movimientos de enero–marzo | 49,525 |
| Acumulaciones / canjes de todo el período | 47,030 / 2,495 |
| Puntos emitidos en marzo, política fija | 4,080,306 |
| Puntos emitidos en marzo, política dinámica | 4,271,226 |
| Emisión adicional | 190,920 (+4.68 %) |
| Puntos canjeados en marzo, ambas políticas | 1,417,116 |
| Clientes con puntos extra / activos en marzo | 605 / 907 |
| Saldo final mínimo, ambas políticas | 150 |

Las gráficas muestran tamaños de segmento, rezagos, afinidades y emisión por segmento/producto. En riesgo concentra el 38.73 % de los puntos adicionales. Esta emisión es una medida en puntos: no equivale a costo en quetzales ni a ventas incrementales.

Validamos los cálculos, la selección de reglas, la cobertura de líneas, la vigencia, el redondeo, los topes y los saldos cronológicos. Comprobamos que los Parquet conserven el esquema y el contenido de las tablas y que cada línea corresponda a un único movimiento.

## Alcance de la reproducción

La POC parte de saldo cero y usa un catálogo sintético fijo; acredita al terminar cada compra. No implementa topes diarios, caducidad, demora de acreditación, devoluciones ni todas las restricciones del programa real. Los tres meses de 2025 son fechas de simulación. La comparación conserva las compras y los canjes originales, sin simular una respuesta comercial a los incentivos.

Calculamos los perfiles y las afinidades con clientes que tienen historial en la ventana y categorías que presentan ventas. El escenario cumple estas condiciones; el tratamiento de clientes nuevos o categorías sin compras queda fuera de esta simulación.

El entorno de validación utiliza Python 3.12, DuckDB 1.5.5 y scikit-learn 1.9.0. La segmentación coincide con los resultados de referencia obtenidos con scikit-learn 1.9.1. Las semillas y el orden de ejecución permiten reproducir el escenario; `requirements.txt` no fija versiones exactas, por lo que la representación binaria de Parquet puede variar entre entornos.

## Versionado de datos (DVC)

Evaluamos DVC para relacionar versiones de código y datos, reproducir etapas y compartir distintas simulaciones mediante almacenamiento remoto. La propuesta incluye la conservación de perfiles, afinidades, reglas y movimientos asociados a cada escenario.

Mantenemos los datos sintéticos en Git y planteamos DVC como una extensión de MLOps. Para su integración por etapas, proponemos agrupar la preparación de perfiles y la segmentación en una etapa Gold/ML, ya que ambas actualizan el mismo archivo. El almacenamiento compartido permite distribuir las versiones entre los integrantes del equipo.

## Entregables

- [Informe del sistema](doc.md): arquitectura, modelo de datos, análisis, conclusiones, referencias e investigación DVC.
- Réplica local ejecutable: notebooks 01–05 y sus archivos de datos.
- Integración ML: notebook 04; aplicación de sus resultados y comparación: notebook 05.
- Pipeline de scikit-learn empaquetado con entry points: `pyproject.toml` y `src/segmentacion_clientes/`.
- Pruebas adicionales: `tests/test_silver.py` y `tests/test_segmentacion.py`.
- Repositorio: [SergioAle210/ML-Engineering](https://github.com/SergioAle210/ML-Engineering).

Planteamos el piloto con grupo de control, la gestión de modelos con MLflow, la migración a Databricks y la adopción de DVC como extensiones para una operación con datos reales.
