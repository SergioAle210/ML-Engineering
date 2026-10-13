"""Converts a notebook to HTML with nbconvert, then prints that HTML to PDF with headless Chromium."""
import sys
from pathlib import Path

from nbconvert import HTMLExporter
from playwright.sync_api import sync_playwright

PRINT_STYLE = """
<style>
@media print {
  body { padding: 0 !important; }
  .jp-Notebook { padding: 0 !important; }
  .jp-OutputArea-output, .jp-RenderedImage img { max-width: 100% !important; }
  .jp-Cell { page-break-inside: avoid; }
  pre { white-space: pre-wrap !important; word-break: break-word; }
}
</style>
"""
PDF_MARGIN = {"top": "14mm", "bottom": "14mm", "left": "12mm", "right": "12mm"}


def notebook_to_html(notebook_path):
    body, _ = HTMLExporter(template_name="lab").from_filename(str(notebook_path))
    html_path = notebook_path.with_suffix(".html")
    html_path.write_text(body.replace("</head>", PRINT_STYLE + "</head>", 1), encoding="utf-8")
    return html_path


def html_to_pdf(html_path):
    pdf_path = html_path.with_suffix(".pdf")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.goto(html_path.as_uri())
        page.wait_for_timeout(1500)
        page.pdf(path=str(pdf_path), format="Letter", margin=PDF_MARGIN, print_background=True)
        browser.close()
    return pdf_path


if __name__ == "__main__":
    html = notebook_to_html(Path(sys.argv[1]).resolve())
    print("html:", html)
    print("pdf:", html_to_pdf(html))
