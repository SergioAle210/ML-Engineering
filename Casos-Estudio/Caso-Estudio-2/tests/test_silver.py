"""Pruebas de integridad para el notebook 02, sin modificar los datos del proyecto."""

import copy
import csv
import tempfile
import unittest
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "02_bronze_a_silver.ipynb"
namespace = {}
for cell in nbformat.read(NOTEBOOK, as_version=4).cells:
    if cell.cell_type == "code" and "silver-definitions" in cell.metadata.get("tags", []):
        exec(compile(cell.source, str(NOTEBOOK), "exec"), namespace)

SCHEMAS = namespace["SCHEMAS"]
clean_bronze = namespace["clean_bronze"]
validate_silver = namespace["validate_silver"]
export_silver = namespace["export_silver"]


def fixture():
    values = {
        "clientes": [
            [1, "Cliente A", "a@example.com", "2024-12-01"],
            [2, "Cliente B", "b@example.com", "2024-12-01"],
        ],
        "productos": [
            [1, "Producto A", "bebidas", "10.00", 100],
            [2, "Producto B", "postres", "10.00", 100],
        ],
        "restaurantes": [[1, "Restaurante", "Zona 1"]],
        "compras": [
            [1, 1, 1, "2025-01-01 10:00:00", "20.00", "efectivo"],
            [2, 1, 1, "2025-01-02 10:00:00", "10.00", "tarjeta"],
            [3, 1, 1, "2025-01-03 10:00:00", "0.00", "efectivo"],
        ],
        "lineas_compra": [
            [1, 1, 1, 1, "10.00", "10.00", "false"],
            [2, 1, 2, 1, "10.00", "10.00", "false"],
            [3, 2, 1, 1, "10.00", "10.00", "false"],
            [4, 3, 2, 1, "0.00", "0.00", "true"],
        ],
    }
    return {table: [dict(zip(SCHEMAS[table], row)) for row in rows] for table, rows in values.items()}


class SilverTests(unittest.TestCase):
    def run_case(self, tables):
        # Incluso los fixtures temporales se mantienen dentro de la carpeta actual.
        directory = tempfile.TemporaryDirectory(prefix=".validacion-silver-", dir=ROOT)
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        for table, rows in tables.items():
            with (path / f"{table}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=SCHEMAS[table])
                writer.writeheader()
                writer.writerows(rows)
        connection = clean_bronze(path)
        self.addCleanup(connection.close)
        validate_silver(connection)
        return connection, path

    def ids(self, connection, table):
        return [row[0] for row in connection.execute(f"SELECT id FROM silver_{table} ORDER BY id").fetchall()]

    def test_duplicates_normalization_and_zero_total_redemption(self):
        tables = fixture()
        tables["clientes"][0]["correo"] = " A@EXAMPLE.COM "
        for rows in tables.values():
            rows.append(copy.deepcopy(rows[0]))
        connection, path = self.run_case(tables)
        self.assertEqual(self.ids(connection, "compras"), [1, 2, 3])
        self.assertEqual(self.ids(connection, "lineas_compra"), [1, 2, 3, 4])
        self.assertEqual(connection.execute("SELECT correo FROM silver_clientes WHERE id=1").fetchone()[0], "a@example.com")
        self.assertTrue(all(row["cuarentena"] == 1 for row in validate_silver(connection)))
        export_silver(connection, path / "silver", path / "cuarentena")
        # Reejecutar la exportación reemplaza, sin acumular filas.
        export_silver(connection, path / "silver", path / "cuarentena")
        with (path / "cuarentena/clientes.csv").open(encoding="utf-8") as stream:
            rejected = list(csv.DictReader(stream))
        self.assertEqual(rejected[0]["correo"], " A@EXAMPLE.COM ")
        self.assertEqual(rejected[0]["motivos"], "duplicado_exacto")

    def test_invalid_line_rejects_entire_purchase(self):
        for changes in [
            {"cantidad": ""}, {"cantidad": "1.5"}, {"precio_unitario": "10.001"},
            {"pagado_con_puntos": "quizas"}, {"producto_id": 999},
            {"precio_unitario": "-10.00", "subtotal": "-10.00"},
            {"subtotal": "9.00"}, {"pagado_con_puntos": "true"},
        ]:
            with self.subTest(changes=changes):
                tables = fixture()
                tables["lineas_compra"][0].update(changes)
                connection, _ = self.run_case(tables)
                self.assertEqual(self.ids(connection, "compras"), [2, 3])
                self.assertEqual(self.ids(connection, "lineas_compra"), [3, 4])

    def test_invalid_header_propagates_to_lines(self):
        for changes in [
            {"cliente_id": 999}, {"restaurante_id": 999}, {"forma_pago": ""},
            {"fecha_hora": "2025-02-30 10:00:00"}, {"fecha_hora": "2025-01-01 23:00:00"},
            {"fecha_hora": "2024-11-01 10:00:00"}, {"monto_total": "19.00"},
        ]:
            with self.subTest(changes=changes):
                tables = fixture()
                tables["compras"][0].update(changes)
                connection, _ = self.run_case(tables)
                self.assertEqual(self.ids(connection, "compras"), [2, 3])
                self.assertEqual(self.ids(connection, "lineas_compra"), [3, 4])

    def test_conflicting_ids_are_not_silently_selected(self):
        tables = fixture()
        conflict = dict(tables["clientes"][0], nombre="Otra persona")
        tables["clientes"].append(conflict)
        connection, _ = self.run_case(tables)
        self.assertEqual(self.ids(connection, "clientes"), [2])
        self.assertEqual(self.ids(connection, "compras"), [])
        self.assertEqual(self.ids(connection, "lineas_compra"), [])

    def test_shared_email_rejects_both_customer_ids(self):
        tables = fixture()
        tables["clientes"][1]["correo"] = " A@EXAMPLE.COM "
        connection, _ = self.run_case(tables)
        self.assertEqual(self.ids(connection, "clientes"), [])
        self.assertEqual(self.ids(connection, "compras"), [])

    def test_rejected_product_propagates_to_related_purchases(self):
        tables = fixture()
        tables["productos"][1]["precio_normal"] = "-1.00"
        connection, _ = self.run_case(tables)
        self.assertEqual(self.ids(connection, "productos"), [1])
        self.assertEqual(self.ids(connection, "compras"), [2])
        self.assertEqual(self.ids(connection, "lineas_compra"), [3])

    def test_orphan_line_and_empty_purchase(self):
        tables = fixture()
        tables["lineas_compra"][0]["compra_id"] = 999
        tables["lineas_compra"] = [row for row in tables["lineas_compra"] if row["id"] != 3]
        connection, _ = self.run_case(tables)
        self.assertEqual(self.ids(connection, "compras"), [3])
        self.assertEqual(self.ids(connection, "lineas_compra"), [4])


if __name__ == "__main__":
    unittest.main()
