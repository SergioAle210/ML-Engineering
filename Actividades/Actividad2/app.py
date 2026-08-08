"""Streamlit dashboard for the gasolina price dataset.

Run locally with: streamlit run app.py
Deployable as-is (e.g. Streamlit Community Cloud): it only reads
data/processed/gasolina_dataset.csv, produced offline by src/pipeline.py.
The OCR pipeline itself needs macOS (Vision framework) to run, but this
dashboard has no such dependency -- ship the CSV alongside the app.
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.pipeline import save_manual_override

ROOT = Path(__file__).resolve().parent
DATASET_CSV = ROOT / "data" / "processed" / "gasolina_dataset.csv"
CONVERTED_DIR = ROOT / "data" / "interim" / "converted"

st.set_page_config(page_title="Bitácora de Gasolina", page_icon="⛽", layout="wide")

ACCENT = "#b8792b"
GOOD = "#2e7d5b"
WARN = "#b3721f"


@st.cache_data
def load_data(path: Path, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["datetime_original"], format="%Y:%m:%d %H:%M:%S")
    return df.sort_values("dt")


if not DATASET_CSV.exists():
    st.error(f"No dataset found at `{DATASET_CSV}`. Run `python -m src.pipeline` first.")
    st.stop()

df = load_data(DATASET_CSV, DATASET_CSV.stat().st_mtime)
valid = df[df["parse_ok"]].copy()

if valid.empty:
    st.warning("No readings have been resolved yet (all rows still need review).")
    st.stop()

latest = valid.iloc[-1]
first = valid.iloc[0]
delta = latest["price_per_gallon_gtq"] - first["price_per_gallon_gtq"]
delta_pct = delta / first["price_per_gallon_gtq"] * 100

station = latest.get("geo_station_name") or "—"
road = latest.get("geo_road") or ""
city = latest.get("geo_city") or ""
address = ", ".join(p for p in [road, city] if p)

st.title(f"⛽ Bitácora de Gasolina Súper — {station}")
st.caption(f"{address} · {len(df)} fotos procesadas · {first['dt']:%d %b %Y} → {latest['dt']:%d %b %Y} · Q150.00 fijos por carga")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Precio actual", f"Q{latest['price_per_gallon_gtq']:.2f}/gal", f"{delta:+.2f} ({delta_pct:+.1f}%)")
col2.metric("Promedio del período", f"Q{valid['price_per_gallon_gtq'].mean():.2f}/gal")
col3.metric("Galones por carga", f"{latest['gallons']:.2f} gal", help="Siempre Q150.00 por carga")
n_auto = (df["data_source"] == "ocr").sum()
col4.metric("Calidad de datos", f"{n_auto}/{len(df)} auto", help="Resto verificado manualmente")

st.subheader("Precio por galón en el tiempo")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=valid["dt"], y=valid["price_per_gallon_gtq"],
    mode="lines+markers",
    line=dict(color=ACCENT, width=2.5, shape="spline"),
    marker=dict(size=7, color=ACCENT),
    fill="tozeroy", fillcolor="rgba(184,121,43,0.12)",
    hovertemplate="%{x|%d %b %Y}<br>Q%{y:.2f}/gal<extra></extra>",
    name="Súper",
))
fig.update_layout(
    height=360, margin=dict(l=10, r=10, t=10, b=10),
    yaxis=dict(title="Q/gal", tickprefix="Q", range=[
        valid["price_per_gallon_gtq"].min() * 0.95,
        valid["price_per_gallon_gtq"].max() * 1.05,
    ]),
    xaxis=dict(title=None),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    showlegend=False,
)
st.plotly_chart(fig, width="stretch")

left, right = st.columns([1.6, 1])

with left:
    st.subheader("Lecturas")
    display_cols = {
        "dt": "Fecha y hora",
        "total_gtq": "Total (Q)",
        "gallons": "Galones",
        "price_per_gallon_gtq": "Q/gal",
        "data_source": "Origen",
        "filename": "Archivo",
    }
    table = df.sort_values("dt", ascending=False)[list(display_cols)].rename(columns=display_cols)
    table["Origen"] = table["Origen"].map({"ocr": "Auto (OCR)", "manual_override": "Revisado"})
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "Fecha y hora": st.column_config.DatetimeColumn(format="DD MMM YYYY, HH:mm"),
            "Total (Q)": st.column_config.NumberColumn(format="Q%.2f"),
            "Galones": st.column_config.NumberColumn(format="%.3f"),
            "Q/gal": st.column_config.NumberColumn(format="Q%.4f"),
        },
    )

with right:
    st.subheader("Cola de revisión")
    flagged = df[df["needs_review"] & (df["data_source"] != "manual_override")]
    if flagged.empty:
        st.success("Nada pendiente — todas las lecturas están verificadas.")
    else:
        for _, r in flagged.iterrows():
            st.warning(f"**{r['filename']}** — {r['dt']:%d %b %Y}: {r['review_reason']}")

    overridden = df[df["data_source"] == "manual_override"]
    if not overridden.empty:
        with st.expander(f"Revisadas manualmente ({len(overridden)})"):
            for _, r in overridden.iterrows():
                st.markdown(f"**{r['filename']}** — {r['dt']:%d %b %Y}")
                st.caption(r["override_note"])

    st.subheader("Ubicación y equipo")
    st.markdown(f"""
    - **Estación:** {station}
    - **Dirección:** {latest.get('geo_display_name', '—')}
    - **Coordenadas:** `{latest['gps_latitude']:.5f}, {latest['gps_longitude']:.5f}`
    - **Altitud:** {latest['gps_altitude_m']:.0f} m
    - **Cámara:** {latest['camera_model']}
    """)
    st.map(pd.DataFrame({"lat": [latest["gps_latitude"]], "lon": [latest["gps_longitude"]]}), size=20)

st.divider()
st.header("Revisión manual", anchor=False)
st.caption(
    "El OCR no siempre está seguro de lo que leyó (brillos, reflejos, dígitos ambiguos). "
    "Revisá la foto contra el valor extraído y corregilo si hace falta — la corrección queda "
    "guardada en `data/manual_overrides.csv` y mejora la base para la predicción futura."
)

review_options = df.sort_values("dt", ascending=False)
n_pending = int((review_options["needs_review"] & (review_options["data_source"] != "manual_override")).sum())

if n_pending:
    st.badge(f"{n_pending} pendiente(s) de revisar", icon=":material/warning:", color="orange")
else:
    st.badge("Todo revisado", icon=":material/check_circle:", color="green")


def _row_label(row) -> str:
    flag = " · ⚠️ necesita revisión" if row["needs_review"] and row["data_source"] != "manual_override" else ""
    return f"{row['dt']:%d %b %Y, %H:%M} — {row['filename']}{flag}"


selected_label = st.selectbox(
    "Foto a revisar",
    options=[_row_label(r) for _, r in review_options.iterrows()],
    label_visibility="collapsed",
)
selected_row = review_options.iloc[[_row_label(r) for _, r in review_options.iterrows()].index(selected_label)]

img_col, form_col = st.columns([1.2, 1])

with img_col:
    jpeg_path = CONVERTED_DIR / f"{Path(selected_row['filename']).stem}.jpg"
    if jpeg_path.exists():
        st.image(str(jpeg_path), width="stretch", caption=selected_row["filename"])
    else:
        st.info(f"No se encontró la imagen convertida en `{jpeg_path}`.")

with form_col:
    st.markdown(f"**Origen del dato actual:** {'Auto (OCR)' if selected_row['data_source'] == 'ocr' else 'Revisado manualmente'}")
    if selected_row["needs_review"] and selected_row["data_source"] != "manual_override":
        st.warning(f"Motivo de revisión: {selected_row['review_reason']}")
    if pd.notna(selected_row.get("total_raw")) or pd.notna(selected_row.get("gallons_raw")):
        st.caption(f"Texto crudo leído por OCR — total: `{selected_row.get('total_raw', '—')}` · galones: `{selected_row.get('gallons_raw', '—')}`")
    if pd.notna(selected_row.get("override_note")) and selected_row.get("override_note"):
        st.caption(f"Nota de la revisión previa: {selected_row['override_note']}")

    with st.form("manual_review_form", border=False):
        c1, c2 = st.columns(2)
        total_input = c1.number_input(
            "Total (Q)",
            min_value=0.0,
            value=float(selected_row["total_gtq"]) if pd.notna(selected_row["total_gtq"]) else 150.0,
            step=0.01,
            format="%.2f",
        )
        gallons_input = c2.number_input(
            "Galones",
            min_value=0.0,
            value=float(selected_row["gallons"]) if pd.notna(selected_row["gallons"]) else 0.0,
            step=0.001,
            format="%.3f",
        )
        note_input = st.text_area(
            "Nota (opcional)",
            placeholder="Ej: confirmado visualmente, el dígito de las milésimas estaba tapado por brillo",
        )
        submitted = st.form_submit_button("Guardar corrección", icon=":material/save:", type="primary")

    if submitted:
        if gallons_input <= 0:
            st.error("Los galones deben ser mayores a 0.")
        else:
            save_manual_override(
                filename=selected_row["filename"],
                total_gtq=total_input,
                gallons=gallons_input,
                note=note_input or "Confirmado manualmente desde el dashboard",
            )
            st.cache_data.clear()
            st.success(f"Guardado. {selected_row['filename']} ahora usa el valor revisado.")
            st.rerun()

st.divider()
st.caption("Pipeline: `src/pipeline.py` → `data/processed/gasolina_dataset.csv` → esta app. "
           "Agrega fotos nuevas a `data/raw/`, corre `python -m src.pipeline`, y recarga.")
