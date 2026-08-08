# Bitácora de precios de gasolina

Pipeline que extrae datos de fotos del surtidor (total pagado, galones, fecha,
ubicación GPS, precio por galón) y los muestra en un dashboard de Streamlit.

Este proyecto sigue las primeras tres fases de CRISP-DM: entendimiento del
negocio, entendimiento de los datos y preparación de los datos. El resultado
de las tres fases es el dataset limpio en `data/processed/gasolina_dataset.csv`.

## Fase 1: Entendimiento del negocio

**Problema:** no tengo visibilidad de cómo varía el precio por galón de
gasolina Súper que compro, siempre en la misma estación (Shell, Carretera a
El Salvador). Cada carga es un monto fijo de Q150.00; lo único que cambia es
cuántos galones entrega ese monto, y por lo tanto el precio implícito por galón.

**Objetivo:** convertir las fotos que tomo del display del surtidor en cada
carga (como comprobante) en una serie de tiempo de precio/galón, sin tener que
transcribir nada a mano.

**Criterio de éxito:** un dataset donde cada fila sea una carga con su precio
por galón, fecha, ubicación y una marca clara de si el dato viene de OCR
automático o fue confirmado a mano — nunca un valor inventado sin marcar.

## Fase 2: Entendimiento de los datos

**Fuentes de datos:**
- Fotos `HEIC` del surtidor (`data/raw/IMG_*.HEIC`) — cada una captura *dos*
  cosas en el mismo encuadre: el display "Esta Venta" (total pagado + galones
  de mi carga, arriba) y, más abajo en la misma foto, el tablero de precios
  por galón de los 4 combustibles de la estación (Diesel/Regular/Súper/
  V-Power). No son fotos separadas — descubrí esto revisando una foto ya
  procesada, así que no hace falta ningún esquema de nombres especial.
- Metadata EXIF/GPS vía `exiftool` (`src/exif_extract.py`) — fecha/hora exacta,
  coordenadas GPS, modelo de cámara.
- Reverse geocoding vía Nominatim (`src/geocode.py`) — nombre y dirección de
  la estación a partir del GPS.

Los 4 precios del tablero son un dato distinto al de mi carga: no es "lo que
pagué", es "lo que costaba en ese momento" cada combustible — útil como
variable externa para un futuro modelo (¿el precio que pago sigue al
posteado?).

**Estado actual del dataset** (5 fotos procesadas, `data/processed/gasolina_dataset.csv`):

| filename | fecha | galones | Q/gal | origen |
|---|---|---|---|---|
| IMG_8271 | 09 jul | 3.938 | 38.09 | OCR |
| IMG_8286 | 15 jul | 3.789 | 39.59 | OCR |
| IMG_8304 | 22 jul | 3.651 | 41.08 | revisado a mano |
| IMG_8322 | 26 jul | 3.564 | 42.09 | revisado a mano |
| IMG_8325 | 30 jul | 3.564 | 42.09 | OCR |

Las 5 filas tienen `parse_ok=True` y 0 pendientes de revisión, pero 2/5 (40%)
necesitaron corrección manual — el motivo real en ambos casos fue brillo/glare
tapando un dígito de galones que el OCR no pudo leer con confianza.

**Riesgos de calidad ya identificados:**
- El total casi siempre se puede leer, pero los galones (más dígitos, más
  sensibles al glare) son el campo que más falla.
- Precisión de OCR distinta por plataforma: macOS (Vision) lee mejor que
  Windows/Linux (Tesseract + modelo de 7 segmentos).
- El monto de Q150.00 es conocido de antemano y sirve como ancla de validación:
  si el OCR no logra leerlo con confianza, se usa ese valor fijo pero la fila
  queda marcada `total_assumed_not_confirmed_by_ocr` hasta revisión.

## Fase 3: Preparación de los datos

Pipeline incremental orquestado por `src/pipeline.py` (`python -m src.pipeline`),
identifica cada foto por nombre de archivo y solo reprocesa lo que cambió:

1. **Conversión** (`src/convert.py`) — HEIC → JPEG con orientación EXIF aplicada.
2. **Extracción EXIF** (`src/exif_extract.py`) — metadata completa cacheada a
   `data/interim/exif/`, luego se seleccionan los campos relevantes.
3. **OCR del display** (`src/ocr_pump.py` + backends) — dispatcher por
   plataforma (Vision en macOS, Tesseract en Windows/Linux) que lee los
   dígitos y los parsea a total/galones.
4. **Geocoding inverso** (`src/geocode.py`) — GPS → nombre de estación y
   dirección, con cache en disco para no repetir llamadas a la API.
5. **Detección de anomalías** (`src/outliers.py`) — marca filas donde el OCR
   falló, el total no cuadra con Q150, o el precio/galón se desvía >15% de la
   mediana del dataset.
6. **Revisión manual** (`app.py`, dashboard Streamlit) — cuando una fila queda
   marcada, se corrige visualmente contra la foto y la corrección se guarda en
   `data/manual_overrides.csv`. `apply_manual_overrides` reconstruye el dataset
   completo desde el baseline OCR en cada corrida, así que es idempotente:
   nunca se acumulan correcciones sobre correcciones.

**Salida:** `data/processed/gasolina_dataset.csv` y `.json` — una fila por
foto, con `data_source` (`ocr` / `manual_override`) y `needs_review` explícitos.

**Precios del tablero** (mismas fotos, columnas `price_diesel_gtq`,
`price_regular_gtq`, `price_super_gtq`, `price_vpower_gtq` en el mismo CSV):
`src/ocr_price_board.py` (dispatcher, solo implementado en macOS/Vision por
ahora — ver docstring de `ocr_backend_tesseract.read_price_board`) ubica cada
LCD por la etiqueta impresa debajo ("Diesel", "Regular", "Super", "V-power"),
no por posición fija, y guarda un recorte de cada una en
`data/interim/price_board_crops/`.

A diferencia del total/galones, estas 4 pantallitas son mucho más chicas y
ruidosas — probado con una foto real, ni Vision ni Tesseract las leen con
confianza (un recorte + OCR devolvió `93,75891` en vez de algo como `31.09`).
Por eso `price_*_gtq` **nunca se llena automáticamente**: el pipeline solo
ubica y recorta cada LCD (`price_*_crop`) y guarda lo que el OCR alcanzó a
adivinar como pista sin confiar en eso (`price_*_raw`). El dato real sale de
`app.py`, confirmando a ojo contra la foto — mismo principio que el resto del
pipeline: nunca inventar un valor que no se pudo confirmar. `board_needs_review`
queda en `True` hasta que se revisa cada foto, y la corrección se guarda en
`data/price_board_overrides.csv`.

Nota: en fotos con ángulo distinto el recorte automático a veces cae sobre
texto vecino en vez de la pantalla correcta (offsets calibrados a mano contra
una sola foto) — por eso la revisión siempre muestra la foto completa, no solo
el recorte, y el panel de Diesel a veces queda sin ubicar si algo lo tapa
físicamente (el poste del surtidor, en varias de estas fotos).

Todavía no hay fase de modelado: el objetivo actual es solo dejar el dataset
limpio y descargable desde el dashboard (barra lateral) como insumo de un
modelo más adelante.

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Además de las dependencias de Python, hacen falta dos herramientas externas:

| Herramienta | macOS | Linux (Debian/Ubuntu) | Windows |
|---|---|---|---|
| `exiftool` (metadata) | `brew install exiftool` | `apt install libimage-exiftool-perl` | [instalador oficial](https://exiftool.org) |
| `tesseract` (solo Windows/Linux, para OCR) | no hace falta | `apt install tesseract-ocr` | [instalador oficial](https://github.com/UB-Mannheim/tesseract/wiki) |

En macOS el OCR usa el framework **Vision** de Apple (vía `pyobjc`, ya en
`requirements.txt`) y no necesita Tesseract. En Windows/Linux se usa
**Tesseract** con un modelo entrenado para pantallas de 7 segmentos
(`models/letsgodigital.traineddata`, ya incluido en el repo).

## Uso

```bash
# Procesar fotos nuevas en data/raw/*.HEIC -> data/processed/gasolina_dataset.csv
python -m src.pipeline

# Levantar el dashboard (confirmá ahí los precios del tablero pendientes;
# botón de descarga del CSV completo en la barra lateral)
streamlit run app.py
```

## Multiplataforma: qué esperar en cada sistema

El pipeline corre completo en Windows, Linux y macOS, pero **no lee las fotos
con la misma precisión en todos lados**:

- **macOS** (Vision framework): mejor lectura, la mayoría de fotos se
  resuelven solas.
- **Windows/Linux** (Tesseract): motor más débil para estos dígitos de 7
  segmentos fotografiados. Es normal que más fotos terminen en la cola de
  "Revisión manual" del dashboard — el pipeline nunca inventa un valor que no
  pudo confirmar, lo marca para que lo confirmes vos a mano ahí mismo (ver
  sección "Revisión manual" en `app.py`).

En ambos casos, si no se pudo leer el total con confianza, se usa como valor
de referencia el monto fijo conocido de este dataset (Q150.00) pero la fila
queda marcada como no confirmada por OCR hasta que se revise.

El dataset resultante (`data/processed/gasolina_dataset.csv`) y el dashboard
no dependen de ninguna librería específica de plataforma — son portables una
vez generados.
