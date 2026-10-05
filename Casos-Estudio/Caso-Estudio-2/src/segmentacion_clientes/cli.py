"""Entry points de consola de la segmentación (declarados en pyproject.toml)."""

import argparse
import json
from pathlib import Path

import duckdb
import joblib
import pandas as pd
import sklearn

from . import pipeline as seg


def _parser(descripcion):
    parser = argparse.ArgumentParser(description=descripcion)
    parser.add_argument(
        "--datos", type=Path, default=None,
        help="Carpeta data/ con silver/ y gold/ (por defecto, la del caso que contiene el directorio actual).",
    )
    return parser


def _carpeta_datos(args):
    return (args.datos or seg.localizar_proyecto() / "data").resolve()


def _modelo_por_defecto(datos):
    return datos.parent / "models" / "segmentacion_kmeans.joblib"


def diagnostico(argv=None):
    """segmentacion-diagnostico: inercia y silueta para un rango de k."""
    parser = _parser("Diagnóstico del número de grupos de K-means.")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    args = parser.parse_args(argv)

    datos = _carpeta_datos(args)
    connection = seg.conectar_silver(datos / "silver")
    perfil_clientes = seg.leer_perfiles(connection, datos / "gold" / "perfil_clientes.parquet")
    connection.close()

    tabla = seg.diagnosticar_k(perfil_clientes, range(args.k_min, args.k_max + 1))
    print(tabla.to_string())
    print(f"\nMejor silueta: k={tabla['silueta'].idxmax()}; el caso usa k={seg.NUM_SEGMENTOS} por interpretación de negocio.")


def entrenar(argv=None):
    """segmentacion-entrenar: ajusta el pipeline, exporta Gold y guarda el modelo."""
    parser = _parser("Entrena la segmentación K-means y actualiza las tablas de Gold.")
    parser.add_argument("--modelo", type=Path, default=None, help="Ruta del .joblib (por defecto, models/segmentacion_kmeans.joblib).")
    parser.add_argument("--metricas", type=Path, default=None, help="Archivo JSON opcional para guardar las métricas.")
    args = parser.parse_args(argv)

    datos = _carpeta_datos(args)
    modelo = args.modelo or _modelo_por_defecto(datos)
    connection = seg.conectar_silver(datos / "silver")
    perfil_clientes = seg.leer_perfiles(connection, datos / "gold" / "perfil_clientes.parquet")
    print(f"DuckDB {duckdb.__version__} — scikit-learn {sklearn.__version__}")
    print(f"perfil_clientes: {len(perfil_clientes)} clientes")

    pipeline, grupos, metricas = seg.entrenar(perfil_clientes)
    resumen_grupos = seg.resumir_grupos(perfil_clientes, grupos)
    nombres_grupo = seg.nombrar_grupos(resumen_grupos)
    seg.resumir_segmentos(resumen_grupos, nombres_grupo)
    perfil_clientes["segmento"] = pd.Series(grupos).map(nombres_grupo).to_numpy()

    inicio_mes1, fin_ventana = seg.ventana_analisis(connection)
    afinidad = seg.calcular_afinidad(connection, perfil_clientes, inicio_mes1, fin_ventana)
    filas = seg.validar_salidas(connection, perfil_clientes, afinidad, inicio_mes1, fin_ventana)
    seg.exportar_gold(connection, perfil_clientes, afinidad, datos / "gold", filas)
    connection.close()

    modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "nombres_grupo": nombres_grupo,
        "sklearn_version": sklearn.__version__,
    }, modelo)

    conteo = perfil_clientes["segmento"].value_counts().reindex(seg.SEGMENTOS)
    metricas["clientes_por_segmento"] = {s: int(n) for s, n in conteo.items()}
    print(f"Silueta con k={seg.NUM_SEGMENTOS}: {metricas['silueta']:.4f}")
    print(f"ARI frente a otras 10 semillas: mínimo {metricas['ari_minimo']:.4f}, promedio {metricas['ari_promedio']:.4f}")
    print("Clientes por segmento:")
    print(conteo.to_string())
    for name, n in filas.items():
        print(f"{name}: {n} filas exportadas a Gold.")
    print(f"Modelo guardado en {modelo}")
    if args.metricas:
        args.metricas.write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")


def predecir(argv=None):
    """segmentacion-predecir: asigna segmentos con un modelo guardado."""
    parser = _parser("Asigna segmentos a perfiles de clientes con un pipeline entrenado.")
    parser.add_argument("--modelo", type=Path, default=None, help="Ruta del .joblib (por defecto, models/segmentacion_kmeans.joblib).")
    parser.add_argument("--entrada", type=Path, default=None, help="Parquet o CSV de perfiles (por defecto, gold/perfil_clientes.parquet).")
    parser.add_argument("--salida", type=Path, required=True, help="Archivo de salida (.parquet o .csv).")
    args = parser.parse_args(argv)

    datos = _carpeta_datos(args) if args.entrada is None or args.modelo is None else None
    modelo = joblib.load(args.modelo or _modelo_por_defecto(datos))
    entrada = args.entrada or datos / "gold" / "perfil_clientes.parquet"
    lector = pd.read_csv if entrada.suffix == ".csv" else pd.read_parquet
    perfiles = lector(entrada).drop(columns="segmento", errors="ignore")

    resultado = seg.asignar_segmentos(modelo["pipeline"], modelo["nombres_grupo"], perfiles)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    if args.salida.suffix == ".csv":
        resultado.to_csv(args.salida, index=False)
    else:
        resultado.to_parquet(args.salida, index=False)
    print(resultado["segmento"].value_counts().reindex(seg.SEGMENTOS, fill_value=0).to_string())
    print(f"{len(resultado)} perfiles segmentados en {args.salida}")
