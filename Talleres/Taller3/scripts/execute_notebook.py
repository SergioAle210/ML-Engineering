"""Executes a notebook in place and drops MLflow's per-run link lines from its outputs."""
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NOISE_MARKERS = ("View run", "View experiment")


def drop_noise(notebook):
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.outputs:
            if output.get("output_type") != "stream":
                continue
            lines = output["text"].splitlines(keepends=True)
            output["text"] = "".join(line for line in lines if not any(m in line for m in NOISE_MARKERS))
        cell.outputs = [o for o in cell.outputs if o.get("output_type") != "stream" or o["text"].strip()]


def main(path):
    notebook_path = Path(path).resolve()
    notebook = nbformat.read(notebook_path, as_version=4)
    ExecutePreprocessor(timeout=3000, kernel_name="python3").preprocess(
        notebook, {"metadata": {"path": str(notebook_path.parent)}}
    )
    drop_noise(notebook)
    nbformat.write(notebook, notebook_path)


if __name__ == "__main__":
    main(sys.argv[1])
