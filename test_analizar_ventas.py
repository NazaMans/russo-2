import unittest

import pandas as pd

from analizar_ventas import clean_sales


class CleaningTests(unittest.TestCase):
    def test_duplicate_invalid_date_and_unreliable_income(self):
        row = {
            "id_registro": "1", "fecha_venta": "2024-01-03 00:00:00",
            "producto": "Mouse", "categoria": "Accesorios", "region": "ESTE ",
            "cantidad": 2, "precio_unitario": "10", "descuento_pct": 0.1,
            "total_venta": 18, "edad_cliente": 30, "canal_venta": "Online",
            "medio_pago": "Tarjeta", "devolucion": "False",
            "dias_entrega": 2, "satisfaccion": 5,
        }
        duplicate = row.copy()
        contradiction = {**row, "id_registro": "2", "fecha_venta": "FECHA_INVALIDA",
                         "region": None, "total_venta": 999, "precio_unitario": "DATO_INVALIDO"}
        data, audit = clean_sales(pd.DataFrame([row, duplicate, contradiction]))
        self.assertEqual(audit["duplicados_exactos_eliminados"], 1)
        self.assertEqual(audit["fechas_invalidas"], 1)
        self.assertEqual(audit["importes_no_verificables"], 1)
        self.assertEqual(data["total_venta_validado"].sum(), 18)
        self.assertEqual(data.loc[0, "region"], "Este")
        self.assertEqual(data.loc[1, "region"], "Sin dato")
        self.assertTrue(pd.isna(data.loc[1, "fecha_venta"]))

    def test_inconsistent_total_keeps_row_but_not_revenue(self):
        row = {
            "id_registro": "3", "fecha_venta": "2025-01-03 00:00:00",
            "producto": "Tablet", "categoria": "Electronica", "region": "Sur",
            "cantidad": -1, "precio_unitario": 10, "descuento_pct": 0.1,
            "total_venta": 100, "edad_cliente": 150, "canal_venta": "Online",
            "medio_pago": "Tarjeta", "devolucion": "True",
            "dias_entrega": -5, "satisfaccion": 6,
        }
        data, audit = clean_sales(pd.DataFrame([row]))
        self.assertEqual(audit["valores_fuera_de_rango"]["cantidad"], 1)
        self.assertEqual(audit["valores_fuera_de_rango"]["edad_cliente"], 1)
        self.assertFalse(data.loc[0, "importe_verificado"])
        self.assertTrue(pd.isna(data.loc[0, "total_venta_validado"]))
        row["cantidad"] = 2
        data, audit = clean_sales(pd.DataFrame([row]))
        self.assertEqual(audit["importes_inconsistentes"], 1)
        self.assertEqual(len(data), 1)


if __name__ == "__main__":
    unittest.main()
