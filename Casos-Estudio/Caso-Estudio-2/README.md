# POC de puntos dinámicos de MiMcDonald's

Implementación local del plan de [doc.md](doc.md). Las fases 1 y 2 están implementadas: estructura del proyecto y generación de datos sintéticos en Bronze. Los notebooks 02–05 contienen únicamente el alcance pendiente.

## Preparación y ejecución

Desde esta carpeta:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Abrir `notebooks/01_generar_datos.ipynb` y ejecutar todas sus celdas en orden. La generación usa solo la biblioteca estándar de Python; las dependencias preparan el entorno Jupyter y las fases posteriores del pipeline.

## Estructura

```text
data/
  bronze/       # Cinco CSV sintéticos, incluidos errores intencionales
  silver/       # Pendiente: tablas limpias en Parquet
  gold/         # Pendiente: tablas de negocio en Parquet
  cuarentena/   # Pendiente: registros rechazados
notebooks/
  01_generar_datos.ipynb
  02_bronze_a_silver.ipynb
  03_silver_a_gold.ipynb
  04_segmentacion_ml.ipynb
  05_puntos_e_insights.ipynb
requirements.txt
```

## Datos generados

- Semilla `22434`; período fijo del 1 de enero al 31 de marzo de 2025. Enero y febrero son la ventana de análisis; marzo es la ventana de aplicación. Las fechas y horas representan la hora local de Guatemala.
- Base válida de 1,000 clientes ficticios, 28 productos (cuatro por categoría), 10 restaurantes y 20,000 compras, con entre una y cuatro líneas por compra.
- Cuatro perfiles ocultos de generación: Leales, Madrugadores, En riesgo y Ocasionales. Sus etiquetas solo existen en memoria durante la generación y no se exportan como variables para ML.
- Cuatro productos con baja probabilidad de compra y preferencias por categoría diferenciadas por perfil. Los precios y el catálogo son ilustrativos, no precios oficiales.
- Montos calculados en centavos enteros. Cada línea monetaria acumula `ceil(subtotal × 10)` puntos; cada canje cuesta `precio_en_puntos × cantidad` y tiene precio monetario cero. El saldo se revisa antes del canje, sin usar puntos ganados en esa misma compra.
- Se agregan al final 16 filas duplicadas (2 clientes, 2 productos, 2 restaurantes, 5 compras y 5 líneas), una compra y línea con monto negativo, y una compra y línea con producto inexistente. Los dos últimos casos usan identificadores nuevos; así los originales válidos quedan disponibles para la limpieza. Los conteos de los CSV superan por ello los de la base válida.

El notebook valida la base válida antes de introducir errores, vuelve a leer los CSV y comprueba que las anomalías exportadas sean exactamente las esperadas. También muestra el comportamiento de los perfiles y el desempeño de los productos rezagados. Reejecutarlo con la misma configuración sobrescribe únicamente los cinco CSV generados en Bronze, con contenido reproducible. Silver, Gold y cuarentena quedan vacíos hasta sus respectivas fases.

## Continuación del plan

02 limpiará los CSV con DuckDB; 03 calculará perfiles y ventas con los meses 1–2; 04 aplicará K-means y calculará afinidades; 05 generará reglas, movimientos e insights, aplicando los multiplicadores solo al mes 3. El saldo que usa el generador sirve para simular canjes coherentes; la tabla final `movimientos_puntos` se construirá en 05.
