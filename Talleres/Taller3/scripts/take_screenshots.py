"""Captures MLflow UI pages into capturas/ using headless Chromium."""
import json
import logging
import os
from pathlib import Path

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import mlflow
from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:5000"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "capturas"
VIEWPORT = {"width": 1600, "height": 1000}
RENDER_WAIT_MS = 3500

logging.getLogger("mlflow").setLevel(logging.ERROR)
mlflow.set_tracking_uri(BASE_URL)


def top_level_runs(experiment_name):
    runs = mlflow.search_runs(experiment_names=[experiment_name])
    runs = runs[runs["tags.mlflow.parentRunId"].isna()]
    return {row["tags.mlflow.runName"]: row["run_id"] for _, row in runs.iterrows()}


def experiment_id(experiment_name):
    return mlflow.get_experiment_by_name(experiment_name).experiment_id


ASSISTANT_CLOSE_POSITION = (1564, 36)


def dismiss_overlays(page):
    got_it = page.get_by_role("button", name="Got it")
    if got_it.count():
        got_it.first.click()
    if page.get_by_text("MLflow Assistant").count():
        page.mouse.click(*ASSISTANT_CLOSE_POSITION)
    page.wait_for_timeout(800)


def apply_run_filter(page, query):
    search_box = page.get_by_placeholder("metrics.rmse < 1 and params.model = \"tree\"")
    search_box.fill(query)
    search_box.press("Enter")
    page.wait_for_timeout(RENDER_WAIT_MS)


def capture(page, filename, url, full_page=False, tab_text=None, run_filter=None, file_to_open=None):
    page.goto(url)
    page.wait_for_timeout(RENDER_WAIT_MS)
    dismiss_overlays(page)
    if tab_text:
        page.get_by_role("tab", name=tab_text).first.click()
        page.wait_for_timeout(RENDER_WAIT_MS)
    if run_filter:
        apply_run_filter(page, run_filter)
    if file_to_open:
        page.get_by_text(file_to_open, exact=True).first.click()
        page.wait_for_timeout(RENDER_WAIT_MS)
    page.screenshot(path=str(OUTPUT_DIR / filename), full_page=full_page)
    print("saved", filename)


def main():
    apples_id = experiment_id("Apples Demand")
    churn_id = experiment_id("Churn Pipeline")
    apples_runs = top_level_runs("Apples Demand")
    churn_runs = top_level_runs("Churn Pipeline")

    OUTPUT_DIR.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)

        capture(page, "01_apples_lista_de_runs.png", f"{BASE_URL}/#/experiments/{apples_id}/runs", run_filter="metrics.best_rmse > 0")
        capture(page, "02_apples_run_mejorado_detalle.png", f"{BASE_URL}/#/experiments/{apples_id}/runs/{apples_runs['improved_detrended']}")
        capture(
            page,
            "03_apples_run_mejorado_metricas_modelo.png",
            f"{BASE_URL}/#/experiments/{apples_id}/runs/{apples_runs['improved_detrended']}",
            tab_text="Model metrics",
        )
        compared = json.dumps(list(apples_runs.values()))
        capture(
            page,
            "04_apples_comparacion_de_runs.png",
            f"{BASE_URL}/#/compare-runs?runs={compared}&experiments={json.dumps([apples_id])}",
        )
        capture(page, "05_churn_lista_de_runs.png", f"{BASE_URL}/#/experiments/{churn_id}/runs", run_filter="metrics.test_roc_auc > 0")
        capture(page, "06_churn_run_pipeline_detalle.png", f"{BASE_URL}/#/experiments/{churn_id}/runs/{churn_runs['gradient_boosting_default']}")
        capture(
            page,
            "07_apples_run_mejorado_residuales.png",
            f"{BASE_URL}/#/experiments/{apples_id}/runs/{apples_runs['improved_detrended']}/artifacts",
            file_to_open="residuals.png",
        )
        capture(page, "08_churn_grid_search_detalle.png", f"{BASE_URL}/#/experiments/{churn_id}/runs/{churn_runs['grid_search_gradient_boosting']}")
        browser.close()


if __name__ == "__main__":
    main()
