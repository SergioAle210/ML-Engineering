"""Pruebas de los entry points de segmentación, sobre una copia temporal de data/."""

import shutil
import tempfile
import unittest
from importlib.metadata import entry_points
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from segmentacion_clientes import cli, pipeline as seg


class SegmentacionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Los entry points escriben en Gold y models/: se trabaja sobre una copia.
        cls.directory = tempfile.TemporaryDirectory(prefix=".validacion-segmentacion-", dir=ROOT)
        cls.datos = Path(cls.directory.name) / "data"
        shutil.copytree(ROOT / "data" / "silver", cls.datos / "silver")
        shutil.copytree(ROOT / "data" / "gold", cls.datos / "gold")
        cls.modelo = Path(cls.directory.name) / "modelo.joblib"
        cli.entrenar(["--datos", str(cls.datos), "--modelo", str(cls.modelo)])

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_entry_points_declarados(self):
        comandos = {ep.name: ep.value for ep in entry_points(group="console_scripts")}
        for nombre in ("diagnostico", "entrenar", "predecir"):
            self.assertEqual(comandos.get(f"segmentacion-{nombre}"), f"segmentacion_clientes.cli:{nombre}")

    def test_pipeline_unico(self):
        pasos = [nombre for nombre, _ in seg.construir_pipeline().steps]
        self.assertEqual(pasos, ["preprocesamiento", "kmeans"])

    def test_entrenar_reproduce_gold(self):
        for name in ("perfil_clientes", "afinidad_segmento_categoria"):
            pd.testing.assert_frame_equal(
                pd.read_parquet(ROOT / "data" / "gold" / f"{name}.parquet"),
                pd.read_parquet(self.datos / "gold" / f"{name}.parquet"),
            )
        self.assertTrue(self.modelo.is_file())

    def test_predecir_con_modelo_guardado(self):
        salida = Path(self.directory.name) / "prediccion.csv"
        cli.predecir(["--datos", str(self.datos), "--modelo", str(self.modelo), "--salida", str(salida)])
        prediccion = pd.read_csv(salida)
        gold = pd.read_parquet(self.datos / "gold" / "perfil_clientes.parquet")
        self.assertEqual(prediccion["segmento"].tolist(), gold["segmento"].tolist())


if __name__ == "__main__":
    unittest.main()
