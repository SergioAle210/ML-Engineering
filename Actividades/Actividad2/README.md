# Bitácora de precios de gasolina

Pipeline que extrae datos de fotos del surtidor (total pagado, galones, fecha,
ubicación GPS, precio por galón) y los muestra en un dashboard de Streamlit.

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

# Levantar el dashboard
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
