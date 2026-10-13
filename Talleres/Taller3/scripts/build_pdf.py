"""Renders entrega.md to entrega.pdf through pandoc and headless Chromium."""
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "entrega.html"
PDF_PATH = ROOT / "entrega.pdf"
STYLE = """
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 11pt; line-height: 1.5; color: #1a1a1a; max-width: 100%; }
h1 { font-size: 22pt; border-bottom: 2px solid #333; padding-bottom: 6px; }
h2 { font-size: 16pt; margin-top: 28px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
h3 { font-size: 13pt; margin-top: 20px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 10pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 5px 8px; text-align: left; vertical-align: top; }
th { background: #f0f0f0; }
code { background: #f3f3f3; padding: 1px 4px; border-radius: 3px; font-size: 9.5pt; }
pre { background: #f3f3f3; padding: 10px; border-radius: 4px; overflow-x: auto; }
img { max-width: 100%; border: 1px solid #ccc; margin: 6px 0 14px; page-break-inside: avoid; }
figure { margin: 14px 0; page-break-inside: avoid; break-inside: avoid; }
figcaption { margin-bottom: 4px; }
blockquote { border-left: 3px solid #999; margin-left: 0; padding-left: 12px; color: #444; }
a { color: #0b57d0; }
"""


def main():
    css_path = ROOT / "scripts" / "pdf_style.css"
    css_path.write_text(STYLE)
    subprocess.run(
        ["pandoc", "entrega.md", "-f", "gfm", "-t", "html5", "-s",
         "--metadata", "pagetitle=Taller 3: MLflow con scikit-learn", "-c", str(css_path), "-o", str(HTML_PATH)],
        cwd=ROOT, check=True,
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.goto(HTML_PATH.as_uri())
        page.wait_for_timeout(1000)
        page.pdf(path=str(PDF_PATH), format="Letter", margin={"top": "18mm", "bottom": "18mm", "left": "16mm", "right": "16mm"}, print_background=True)
        browser.close()
    HTML_PATH.unlink()
    css_path.unlink()


if __name__ == "__main__":
    main()
