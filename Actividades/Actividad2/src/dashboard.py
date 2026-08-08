"""Builds a self-contained HTML dashboard from data/processed/gasolina_dataset.csv.

Run with: python -m src.dashboard
Regenerates data/processed/dashboard.html from whatever is currently in the
dataset -- re-run any time after the pipeline adds new photos.
"""
import datetime as dt
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET_CSV = ROOT / "data" / "processed" / "gasolina_dataset.csv"
OUT_HTML = ROOT / "data" / "processed" / "dashboard.html"

CHART_W, CHART_H = 920, 300
CHART_PAD_L, CHART_PAD_R, CHART_PAD_T, CHART_PAD_B = 56, 24, 24, 36


def _fmt_gtq(v: float) -> str:
    return f"Q{v:,.2f}"


def _fmt_date(iso_dt: str) -> str:
    d = dt.datetime.strptime(iso_dt, "%Y:%m:%d %H:%M:%S")
    return d.strftime("%d %b %Y")


def _fmt_datetime(iso_dt: str) -> str:
    d = dt.datetime.strptime(iso_dt, "%Y:%m:%d %H:%M:%S")
    return d.strftime("%d %b %Y, %H:%M")


def build_chart_svg(df: pd.DataFrame) -> str:
    valid = df[df["parse_ok"]].copy()
    valid["dt"] = pd.to_datetime(valid["datetime_original"], format="%Y:%m:%d %H:%M:%S")
    valid = valid.sort_values("dt")

    prices = valid["price_per_gallon_gtq"].tolist()
    dates = valid["dt"].tolist()
    n = len(prices)
    if n < 2:
        return "<p class='chart-empty'>Not enough verified readings yet for a trend line.</p>"

    y_min, y_max = min(prices), max(prices)
    y_pad = max((y_max - y_min) * 0.25, 1.0)
    y_lo, y_hi = y_min - y_pad, y_max + y_pad

    plot_w = CHART_W - CHART_PAD_L - CHART_PAD_R
    plot_h = CHART_H - CHART_PAD_T - CHART_PAD_B

    def x_at(i):
        return CHART_PAD_L + (i / (n - 1)) * plot_w

    def y_at(v):
        return CHART_PAD_T + (1 - (v - y_lo) / (y_hi - y_lo)) * plot_h

    points = [(x_at(i), y_at(p)) for i, p in enumerate(prices)]
    path_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    area_d = (
        path_d
        + f" L {points[-1][0]:.1f},{CHART_PAD_T + plot_h:.1f}"
        + f" L {points[0][0]:.1f},{CHART_PAD_T + plot_h:.1f} Z"
    )

    gridlines = []
    grid_n = 4
    for g in range(grid_n + 1):
        gy = CHART_PAD_T + (g / grid_n) * plot_h
        gval = y_hi - (g / grid_n) * (y_hi - y_lo)
        gridlines.append(
            f'<line x1="{CHART_PAD_L}" y1="{gy:.1f}" x2="{CHART_W - CHART_PAD_R}" y2="{gy:.1f}" class="grid-line" />'
            f'<text x="{CHART_PAD_L - 10}" y="{gy + 4:.1f}" class="grid-label" text-anchor="end">Q{gval:.1f}</text>'
        )

    dots = []
    hit_targets = []
    for i, ((x, y), price, date, src) in enumerate(zip(points, prices, dates, valid["data_source"])):
        is_end = i == n - 1
        r = 5 if is_end else 3.5
        cls = "dot dot-end" if is_end else "dot"
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" class="{cls}" />')
        if is_end:
            dots.append(
                f'<text x="{x:.1f}" y="{y - 14:.1f}" class="dot-label" text-anchor="end">'
                f"Q{price:.2f}/gal</text>"
            )
        label = date.strftime("%d %b")
        src_label = "auto" if src == "ocr" else "reviewed"
        hit_targets.append(
            f'<g class="hit-target" tabindex="0">'
            f'<rect x="{x-18:.1f}" y="{CHART_PAD_T}" width="36" height="{plot_h:.1f}" class="hit-rect" />'
            f"<title>{label} — Q{price:.2f}/gal ({src_label})</title>"
            f"</g>"
        )

    x_labels = []
    step = max(1, n // 5)
    for i in range(0, n, step):
        x_labels.append(
            f'<text x="{points[i][0]:.1f}" y="{CHART_H - 10}" class="x-label" text-anchor="middle">'
            f"{dates[i].strftime('%d %b')}</text>"
        )

    return f"""
    <svg viewBox="0 0 {CHART_W} {CHART_H}" class="trend-chart" role="img"
         aria-label="Precio de gasolina Súper por galón a lo largo del tiempo">
      <defs>
        <linearGradient id="areaFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="var(--accent)" stop-opacity="0.28" />
          <stop offset="100%" stop-color="var(--accent)" stop-opacity="0" />
        </linearGradient>
      </defs>
      {''.join(gridlines)}
      <path d="{area_d}" class="area-path" fill="url(#areaFill)" />
      <path d="{path_d}" class="line-path" />
      {''.join(dots)}
      {''.join(x_labels)}
      {''.join(hit_targets)}
    </svg>
    """


def build_table_rows(df: pd.DataFrame) -> str:
    rows = []
    for _, r in df.sort_values("datetime_original", ascending=False).iterrows():
        badge_cls, badge_txt = {
            "ocr": ("badge-auto", "Auto (OCR)"),
            "manual_override": ("badge-review", "Reviewed"),
        }.get(r["data_source"], ("badge-fail", "Failed"))
        price = f"{r['price_per_gallon_gtq']:.4f}" if pd.notna(r["price_per_gallon_gtq"]) else "—"
        gallons = f"{r['gallons']:.3f}" if pd.notna(r["gallons"]) else "—"
        total = _fmt_gtq(r["total_gtq"]) if pd.notna(r["total_gtq"]) else "—"
        note = r["override_note"] if pd.notna(r.get("override_note")) else ""
        rows.append(f"""
        <tr>
          <td>{_fmt_datetime(r['datetime_original'])}</td>
          <td class="num">{total}</td>
          <td class="num">{gallons}</td>
          <td class="num">Q{price}</td>
          <td><span class="badge {badge_cls}">{badge_txt}</span></td>
          <td class="muted">{r['filename']}</td>
          <td class="muted note">{note}</td>
        </tr>""")
    return "".join(rows)


def build_review_queue(df: pd.DataFrame) -> str:
    flagged = df[df["needs_review"] & (df["data_source"] != "manual_override")]
    if flagged.empty:
        return '<p class="empty-state">Nada pendiente — todas las lecturas están verificadas.</p>'
    items = []
    for _, r in flagged.iterrows():
        items.append(f"""
        <li>
          <span class="chip chip-warn">Revisión</span>
          <strong>{r['filename']}</strong>
          <span class="muted">— {_fmt_date(r['datetime_original'])} · {r['review_reason']}</span>
        </li>""")
    return "<ul class='review-list'>" + "".join(items) + "</ul>"


def build_html(df: pd.DataFrame) -> str:
    valid = df[df["parse_ok"]].sort_values("datetime_original")
    latest = valid.iloc[-1]
    first = valid.iloc[0]
    avg_price = valid["price_per_gallon_gtq"].mean()
    delta = latest["price_per_gallon_gtq"] - first["price_per_gallon_gtq"]
    delta_pct = delta / first["price_per_gallon_gtq"] * 100
    n_total = len(df)
    n_auto = (df["data_source"] == "ocr").sum()
    n_reviewed = (df["data_source"] == "manual_override").sum()

    station = latest.get("geo_station_name") or "Shell"
    road = latest.get("geo_road") or ""
    city = latest.get("geo_city") or ""
    address = ", ".join(p for p in [road, city] if p)

    trend_dir = "up" if delta > 0 else ("down" if delta < 0 else "flat")
    trend_symbol = {"up": "▲", "down": "▼", "flat": "▬"}[trend_dir]
    trend_cls = {"up": "trend-up", "down": "trend-down", "flat": "trend-flat"}[trend_dir]

    chart_svg = build_chart_svg(df)
    table_rows = build_table_rows(df)
    review_html = build_review_queue(df)

    generated_at = dt.datetime.now().strftime("%d %b %Y, %H:%M")
    first_date = _fmt_date(first["datetime_original"])
    latest_date = _fmt_date(latest["datetime_original"])

    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>Bitácora de Gasolina — {station}</title>
<style>
{CSS}
</style>
</head>
<body>
<div class="page">

  <header class="masthead">
    <div class="masthead-text">
      <p class="eyebrow">Bitácora de precios · Gasolina Súper</p>
      <h1>{station} <span class="muted-inline">— {address}</span></h1>
      <p class="subhead">{n_total} fotos procesadas · {first_date} → {latest_date} · Q150.00 fijos por carga</p>
    </div>
    <div class="masthead-meta">
      <span class="muted">Generado {generated_at}</span>
    </div>
  </header>

  <section class="hero-stats">
    <div class="stat-tile stat-tile-primary">
      <p class="stat-label">Precio actual</p>
      <p class="stat-value">Q{latest['price_per_gallon_gtq']:.2f}<span class="stat-unit">/gal</span></p>
      <p class="stat-sub {trend_cls}">{trend_symbol} {abs(delta):.2f} ({abs(delta_pct):.1f}%) desde {first_date}</p>
    </div>
    <div class="stat-tile">
      <p class="stat-label">Promedio del período</p>
      <p class="stat-value">Q{avg_price:.2f}<span class="stat-unit">/gal</span></p>
    </div>
    <div class="stat-tile">
      <p class="stat-label">Galones por carga</p>
      <p class="stat-value">{latest['gallons']:.2f}<span class="stat-unit">gal</span></p>
      <p class="stat-sub muted">por Q150.00</p>
    </div>
    <div class="stat-tile">
      <p class="stat-label">Calidad de datos</p>
      <p class="stat-value">{n_auto}<span class="stat-unit">/{n_total} auto</span></p>
      <p class="stat-sub muted">{n_reviewed} revisadas a mano</p>
    </div>
  </section>

  <section class="panel">
    <div class="panel-head">
      <h2>Precio por galón en el tiempo</h2>
      <p class="muted">Gasolina Súper · Quetzales por galón</p>
    </div>
    {chart_svg}
  </section>

  <section class="grid-2">
    <div class="panel">
      <div class="panel-head">
        <h2>Lecturas</h2>
        <p class="muted">Cada fila proviene de una foto en <code>data/raw/</code></p>
      </div>
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Fecha y hora</th>
              <th class="num">Total</th>
              <th class="num">Galones</th>
              <th class="num">Q/gal</th>
              <th>Origen</th>
              <th>Archivo</th>
              <th>Nota</th>
            </tr>
          </thead>
          <tbody>
            {table_rows}
          </tbody>
        </table>
      </div>
    </div>

    <div class="side-col">
      <div class="panel">
        <div class="panel-head">
          <h2>Cola de revisión</h2>
          <p class="muted">Lecturas que el OCR no pudo confirmar solo</p>
        </div>
        {review_html}
      </div>

      <div class="panel">
        <div class="panel-head">
          <h2>Ubicación y equipo</h2>
        </div>
        <dl class="meta-list">
          <div><dt>Estación</dt><dd>{station}</dd></div>
          <div><dt>Dirección</dt><dd>{latest.get('geo_display_name', '—')}</dd></div>
          <div><dt>Coordenadas</dt><dd class="tabular">{latest['gps_latitude']:.5f}, {latest['gps_longitude']:.5f}</dd></div>
          <div><dt>Altitud</dt><dd class="tabular">{latest['gps_altitude_m']:.0f} m</dd></div>
          <div><dt>Cámara</dt><dd>{latest['camera_model']}</dd></div>
          <div><dt>Lente</dt><dd>{latest['lens_model']}</dd></div>
        </dl>
      </div>
    </div>
  </section>

  <footer class="page-footer">
    <p>Pipeline: <code>src/pipeline.py</code> → <code>data/processed/gasolina_dataset.csv</code> → <code>src/dashboard.py</code>.
    Corre de nuevo tras agregar fotos nuevas a <code>data/raw/</code>; solo procesa lo nuevo.</p>
  </footer>

</div>
</body>
</html>"""


CSS = """
:root {
  --bg: #f3f4f1;
  --surface: #ffffff;
  --surface-sunken: #eceee9;
  --border: #d8dad3;
  --ink: #1c2321;
  --ink-secondary: #4b5450;
  --ink-muted: #7c847e;
  --accent: #b8792b;
  --accent-ink: #7a4f1a;
  --teal: #1f6f63;
  --good: #2e7d5b;
  --warn: #b3721f;
  --warn-bg: #f6e6cf;
  --fail: #b23b3b;
  --fail-bg: #f7e0df;
  --font-body: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  --font-mono: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #15181a;
    --surface: #1d2124;
    --surface-sunken: #23272a;
    --border: #33393c;
    --ink: #eceff2;
    --ink-secondary: #b7bec0;
    --ink-muted: #7f8789;
    --accent: #d99a4e;
    --accent-ink: #f0b876;
    --teal: #4fa997;
    --good: #4fae82;
    --warn: #d99a4e;
    --warn-bg: #3a2f1c;
    --fail: #d97070;
    --fail-bg: #3a2323;
  }
}
:root[data-theme="dark"] {
  --bg: #15181a; --surface: #1d2124; --surface-sunken: #23272a; --border: #33393c;
  --ink: #eceff2; --ink-secondary: #b7bec0; --ink-muted: #7f8789;
  --accent: #d99a4e; --accent-ink: #f0b876; --teal: #4fa997;
  --good: #4fae82; --warn: #d99a4e; --warn-bg: #3a2f1c; --fail: #d97070; --fail-bg: #3a2323;
}
:root[data-theme="light"] {
  --bg: #f3f4f1; --surface: #ffffff; --surface-sunken: #eceee9; --border: #d8dad3;
  --ink: #1c2321; --ink-secondary: #4b5450; --ink-muted: #7c847e;
  --accent: #b8792b; --accent-ink: #7a4f1a; --teal: #1f6f63;
  --good: #2e7d5b; --warn: #b3721f; --warn-bg: #f6e6cf; --fail: #b23b3b; --fail-bg: #f7e0df;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: var(--font-body);
  -webkit-font-smoothing: antialiased;
}
.page {
  max-width: 1180px;
  margin: 0 auto;
  padding: 40px 28px 64px;
  display: flex;
  flex-direction: column;
  gap: 28px;
}

.masthead {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 16px;
  flex-wrap: wrap;
  border-bottom: 1px solid var(--border);
  padding-bottom: 20px;
}
.eyebrow {
  margin: 0 0 6px;
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--accent-ink);
  font-weight: 600;
}
h1 {
  margin: 0;
  font-size: 30px;
  font-weight: 650;
  letter-spacing: -0.01em;
  text-wrap: balance;
}
.muted-inline { color: var(--ink-muted); font-weight: 450; }
.subhead { margin: 8px 0 0; color: var(--ink-secondary); font-size: 14px; }
.muted { color: var(--ink-muted); }

.hero-stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
}
.stat-tile {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 18px 20px;
}
.stat-tile-primary { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent) inset; }
.stat-label { margin: 0 0 8px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--ink-muted); }
.stat-value { margin: 0; font-family: var(--font-mono); font-size: 30px; font-weight: 600; font-variant-numeric: tabular-nums; }
.stat-unit { font-size: 14px; color: var(--ink-muted); margin-left: 4px; }
.stat-sub { margin: 8px 0 0; font-size: 13px; font-variant-numeric: tabular-nums; }
.trend-up { color: var(--fail); }
.trend-down { color: var(--good); }
.trend-flat { color: var(--ink-muted); }

.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 22px 24px;
}
.panel-head { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
.panel-head h2 { margin: 0; font-size: 16px; font-weight: 650; }

.trend-chart { width: 100%; height: auto; overflow: visible; }
.grid-line { stroke: var(--border); stroke-width: 1; }
.grid-label { font-family: var(--font-mono); font-size: 11px; fill: var(--ink-muted); }
.x-label { font-family: var(--font-mono); font-size: 11px; fill: var(--ink-muted); }
.line-path { fill: none; stroke: var(--accent); stroke-width: 2.5; stroke-linecap: round; stroke-linejoin: round; }
.area-path { stroke: none; }
.dot { fill: var(--surface); stroke: var(--accent); stroke-width: 2; }
.dot-end { fill: var(--accent); stroke: var(--surface); stroke-width: 2; }
.dot-label { font-family: var(--font-mono); font-size: 13px; font-weight: 600; fill: var(--accent-ink); }
.hit-rect { fill: transparent; }
.hit-target:hover .hit-rect, .hit-target:focus .hit-rect { fill: color-mix(in srgb, var(--accent) 10%, transparent); }
.chart-empty { color: var(--ink-muted); font-size: 14px; }

.grid-2 { display: grid; grid-template-columns: 1.6fr 1fr; gap: 20px; align-items: start; }
.side-col { display: flex; flex-direction: column; gap: 20px; }

.table-scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
thead th {
  text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
  color: var(--ink-muted); font-weight: 600; padding: 0 10px 8px; border-bottom: 1px solid var(--border);
}
tbody td { padding: 9px 10px; border-bottom: 1px solid var(--surface-sunken); vertical-align: top; }
tbody tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: right; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
td.muted { color: var(--ink-muted); font-size: 12.5px; }
td.note { max-width: 260px; }

.badge { display: inline-block; padding: 3px 9px; border-radius: 999px; font-size: 11.5px; font-weight: 600; }
.badge-auto { background: color-mix(in srgb, var(--good) 16%, transparent); color: var(--good); }
.badge-review { background: var(--warn-bg); color: var(--warn); }
.badge-fail { background: var(--fail-bg); color: var(--fail); }

.review-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 10px; }
.review-list li { font-size: 13.5px; display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
.chip { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em; padding: 2px 8px; border-radius: 6px; }
.chip-warn { background: var(--warn-bg); color: var(--warn); }
.empty-state { color: var(--ink-muted); font-size: 13.5px; margin: 0; }

.meta-list { margin: 0; display: flex; flex-direction: column; gap: 10px; }
.meta-list > div { display: flex; justify-content: space-between; gap: 12px; font-size: 13.5px; border-bottom: 1px dashed var(--surface-sunken); padding-bottom: 8px; }
.meta-list dt { color: var(--ink-muted); }
.meta-list dd { margin: 0; text-align: right; font-weight: 500; }
.meta-list dd.tabular { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }

.page-footer { border-top: 1px solid var(--border); padding-top: 16px; font-size: 12.5px; color: var(--ink-muted); }
code { font-family: var(--font-mono); background: var(--surface-sunken); padding: 1px 5px; border-radius: 4px; font-size: 0.92em; }

@media (max-width: 860px) {
  .hero-stats { grid-template-columns: repeat(2, 1fr); }
  .grid-2 { grid-template-columns: 1fr; }
}
"""


def main():
    df = pd.read_csv(DATASET_CSV)
    html = build_html(df)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote dashboard -> {OUT_HTML}")


if __name__ == "__main__":
    main()
