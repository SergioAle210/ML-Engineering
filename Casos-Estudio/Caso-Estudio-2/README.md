# POC de puntos dinámicos de MiMcDonald's

Implementación local del plan de [doc.md](doc.md). Están implementados la estructura del proyecto, la generación de datos sintéticos (notebook 01) y la limpieza Bronze → Silver (notebook 02). Los notebooks 03–05 contienen únicamente el alcance pendiente.

## Preparación y ejecución

Desde esta carpeta:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Abrir `notebooks/01_generar_datos.ipynb` y ejecutar todas sus celdas en orden; después, ejecutar `notebooks/02_bronze_a_silver.ipynb`. La generación usa solo la biblioteca estándar de Python; la limpieza usa DuckDB. Los notebooks pueden iniciarse desde la raíz del proyecto o desde `notebooks/`.

## Estructura

```text
data/
  bronze/       # Cinco CSV sintéticos, incluidos errores intencionales
  silver/       # Cinco tablas limpias en Parquet
  gold/         # Pendiente: tablas de negocio en Parquet
  cuarentena/   # Cinco CSV de rechazos con valores originales y motivos
notebooks/
  01_generar_datos.ipynb
  02_bronze_a_silver.ipynb
  03_silver_a_gold.ipynb
  04_segmentacion_ml.ipynb
  05_puntos_e_insights.ipynb
requirements.txt
tests/
  test_silver.py
```

## Datos generados

- Semilla `22434`; período fijo del 1 de enero al 31 de marzo de 2025. Enero y febrero son la ventana de análisis; marzo es la ventana de aplicación. Las fechas y horas representan la hora local de Guatemala.
- Base válida de 1,000 clientes ficticios, 28 productos (cuatro por categoría), 10 restaurantes y 20,000 compras, con entre una y cuatro líneas por compra.
- Cuatro perfiles ocultos de generación: Leales, Madrugadores, En riesgo y Ocasionales. Sus etiquetas solo existen en memoria durante la generación y no se exportan como variables para ML.
- Cuatro productos con baja probabilidad de compra y preferencias por categoría diferenciadas por perfil. Los precios y el catálogo son ilustrativos, no precios oficiales.
- Montos calculados en centavos enteros. Cada línea monetaria acumula `ceil(subtotal × 10)` puntos; cada canje cuesta `precio_en_puntos × cantidad` y tiene precio monetario cero. El saldo se revisa antes del canje, sin usar puntos ganados en esa misma compra.
- Se agregan al final 16 filas duplicadas (2 clientes, 2 productos, 2 restaurantes, 5 compras y 5 líneas), una compra y línea con monto negativo, y una compra y línea con producto inexistente. Los dos últimos casos usan identificadores nuevos; así los originales válidos quedan disponibles para la limpieza. Los conteos de los CSV superan por ello los de la base válida.

El notebook 01 valida la base válida antes de introducir errores, vuelve a leer los CSV y comprueba que las anomalías exportadas sean exactamente las esperadas. También muestra el comportamiento de los perfiles y el desempeño de los productos rezagados. Reejecutarlo con la misma configuración sobrescribe únicamente los cinco CSV generados en Bronze, con contenido reproducible. Si cambia Bronze, se debe ejecutar de nuevo el notebook 02 para actualizar Silver y cuarentena.

## Limpieza Bronze → Silver

El notebook 02 convierte tipos, elimina copias duplicadas y valida campos obligatorios, dominios, claves, referencias, fechas e importes con SQL en DuckDB. Los importes se guardan como `DECIMAL(12,2)`; los valores con precisión excesiva se rechazan sin redondearlos. Los duplicados exactos se comparan después de normalizar espacios y tipos; se conserva el primero. Si un ID tiene contenidos distintos, todas sus versiones van a cuarentena. Los correos compartidos entre clientes distintos también se rechazan.

Se conserva la integridad de cada factura: una línea inválida (excepto una copia exacta), un detalle ausente o repetido, o un total que no coincide con la suma de las líneas provoca el rechazo de la compra completa. Los canjes tienen importe cero; una compra formada solo por canjes puede tener total cero. Se valida el precio efectivamente cobrado sin exigir que coincida con el precio actual del catálogo.

Cada CSV de cuarentena incluye las columnas originales, `archivo_origen`, `fila_csv` y `motivos`. La fila indica el orden del registro CSV contando el encabezado; no necesariamente la línea física si un campo contiene saltos de línea. Una fila puede tener varios motivos, pero se cuenta una sola vez en el resumen.

| Tabla | Bronze | Silver | Cuarentena |
|---|---:|---:|---:|
| clientes | 1,002 | 1,000 | 2 |
| productos | 30 | 28 | 2 |
| restaurantes | 12 | 10 | 2 |
| compras | 20,007 | 20,000 | 7 |
| lineas_compra | 49,532 | 49,525 | 7 |

Los 20 rechazos son los 16 duplicados y las dos compras inválidas con sus líneas. La ejecución verifica `Bronze = Silver + cuarentena`, las relaciones y los totales, la conservación de tipos y valores al releer Parquet y que los CSV de Bronze no cambien. Reejecutar reemplaza los mismos cinco Parquet y cinco CSV de cuarentena; Gold sigue pendiente.

Las pruebas adicionales usan datos pequeños en carpetas temporales dentro de este proyecto, que se eliminan al terminar:

```bash
python -m unittest discover -s tests -v
```

## Continuación del plan

03 calculará perfiles y ventas con los meses 1–2; 04 aplicará K-means y calculará afinidades; 05 generará reglas, movimientos e insights, aplicando los multiplicadores solo al mes 3. El saldo que usa el generador sirve para simular canjes coherentes; la tabla final `movimientos_puntos` se construirá en 05.
