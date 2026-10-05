"""Pipeline de scikit-learn y cálculos de la segmentación de clientes (etapa 04)."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 22434
NUM_SEGMENTOS = 4
N_INIT = 20
UMBRAL_AFINIDAD = 1.2
FRANJAS = ["mañana", "almuerzo", "tarde", "noche"]
SEGMENTOS = ["Leales", "Madrugadores", "En riesgo", "Ocasionales"]
VARIABLES_NUMERICAS = ["recencia_dias", "frecuencia_mensual", "gasto_promedio"]
# Ponderar el bloque one-hot de franja como una sola variable.
PESO_FRANJA = 1 / np.sqrt(len(FRANJAS))
COLUMNAS_PERFIL = ["cliente_id", *VARIABLES_NUMERICAS, "franja_preferida", "categoria_preferida", "segmento"]


def sql_text(value):
    return "'" + str(value).replace("'", "''") + "'"


def verificar(condicion, mensaje):
    if not condicion:
        raise ValueError(mensaje)


# --- Datos ---------------------------------------------------------------------------------

def localizar_proyecto(inicio=None):
    """Busca la carpeta del caso (la que contiene doc.md) desde `inicio` hacia arriba."""
    inicio = Path(inicio or Path.cwd()).resolve()
    for carpeta in (inicio, *inicio.parents):
        if (carpeta / "doc.md").is_file():
            return carpeta
    raise FileNotFoundError("No se encuentra la carpeta del caso. Inicie la ejecución desde Caso-Estudio-2.")


def conectar_silver(silver):
    connection = duckdb.connect(":memory:")
    connection.execute("SET threads = 1")
    for table in ("clientes", "productos", "compras", "lineas_compra"):
        parquet = sql_text(Path(silver) / f"{table}.parquet")
        connection.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet({parquet})")
    return connection


def leer_perfiles(connection, ruta):
    """Lee perfil_clientes; al reejecutar se descarta el segmento previo para recalcularlo."""
    return connection.execute(
        f"SELECT * FROM read_parquet({sql_text(ruta)}) ORDER BY cliente_id"
    ).df().drop(columns="segmento", errors="ignore")


# --- Modelo --------------------------------------------------------------------------------

def construir_preprocesamiento():
    return ColumnTransformer(
        [
            ("numericas", StandardScaler(), VARIABLES_NUMERICAS),
            ("franja", Pipeline([
                ("one_hot", OneHotEncoder(categories=[FRANJAS], sparse_output=False)),
                ("escala", StandardScaler()),
            ]), ["franja_preferida"]),
        ],
        transformer_weights={"numericas": 1.0, "franja": PESO_FRANJA},
    )


def construir_pipeline(n_clusters=NUM_SEGMENTOS, random_state=SEED, n_init=N_INIT):
    """Preprocesamiento y K-means en un único sklearn.pipeline.Pipeline."""
    return Pipeline([
        ("preprocesamiento", construir_preprocesamiento()),
        ("kmeans", KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)),
    ])


def diagnosticar_k(perfil_clientes, k_valores=range(2, 9), random_state=SEED):
    """Inercia y silueta para cada número de grupos."""
    X = construir_preprocesamiento().fit_transform(perfil_clientes)
    filas = []
    for k in k_valores:
        modelo_k = KMeans(n_clusters=k, n_init=N_INIT, random_state=random_state).fit(X)
        filas.append({
            "k": k,
            "inercia": round(modelo_k.inertia_, 1),
            "silueta": round(silhouette_score(X, modelo_k.labels_), 4),
        })
    return pd.DataFrame(filas).set_index("k")


def entrenar(perfil_clientes, semillas_estabilidad=range(1, 11)):
    """Ajusta el pipeline y devuelve (pipeline, grupos, métricas)."""
    pipeline = construir_pipeline()
    grupos = pipeline.fit_predict(perfil_clientes)
    X = pipeline[:-1].transform(perfil_clientes)
    estabilidad = [
        adjusted_rand_score(grupos, KMeans(n_clusters=NUM_SEGMENTOS, n_init=N_INIT, random_state=semilla).fit_predict(X))
        for semilla in semillas_estabilidad
    ]
    metricas = {
        "silueta": float(silhouette_score(X, grupos)),
        "ari_minimo": float(min(estabilidad)),
        "ari_promedio": float(np.mean(estabilidad)),
    }
    verificar(not any("categoria" in c for c in pipeline[:-1].get_feature_names_out()), "categoria_preferida no debe entrar al modelo")
    verificar(len(set(grupos)) == NUM_SEGMENTOS, "K-means debe producir cuatro grupos no vacíos")
    verificar(metricas["ari_minimo"] > 0.9, f"Segmentación inestable entre semillas (ARI mínimo {metricas['ari_minimo']:.3f})")
    return pipeline, grupos, metricas


# --- Interpretación ------------------------------------------------------------------------

def resumir_grupos(perfil_clientes, grupos):
    perfil_grupos = perfil_clientes.assign(grupo=grupos)
    resumen = perfil_grupos.groupby("grupo").agg(
        clientes=("cliente_id", "size"),
        recencia_dias=("recencia_dias", "mean"),
        frecuencia_mensual=("frecuencia_mensual", "mean"),
        gasto_promedio=("gasto_promedio", "mean"),
    )
    pct_franja = (
        pd.crosstab(perfil_grupos["grupo"], perfil_grupos["franja_preferida"], normalize="index")
        .reindex(columns=FRANJAS, fill_value=0)
        .add_prefix("pct_")
    )
    categoria_comun = perfil_grupos.groupby("grupo")["categoria_preferida"].agg(
        lambda categorias: categorias.value_counts().index[0]
    ).rename("categoria_mas_comun")
    return resumen.join(pct_franja).join(categoria_comun).round(2)


def nombrar_grupos(resumen):
    pendientes = set(resumen.index)

    def elegir(columna, criterio):
        valores = resumen.loc[sorted(pendientes), columna]
        grupo = valores.idxmin() if criterio == "min" else valores.idxmax()
        pendientes.remove(grupo)
        return grupo

    nombres = {}
    nombres[elegir("frecuencia_mensual", "min")] = "Ocasionales"
    nombres[elegir("pct_mañana", "max")] = "Madrugadores"
    nombres[elegir("recencia_dias", "max")] = "En riesgo"
    nombres[pendientes.pop()] = "Leales"
    return {int(grupo): nombre for grupo, nombre in nombres.items()}


def resumir_segmentos(resumen_grupos, nombres_grupo):
    """Resumen por segmento y comprobación de que cada grupo coincide con su interpretación."""
    verificar(sorted(nombres_grupo.values()) == sorted(SEGMENTOS), "Cada segmento debe asignarse a un solo grupo")
    resumen = resumen_grupos.rename(index=nombres_grupo).rename_axis("segmento").loc[SEGMENTOS]
    leales, madrugadores, riesgo, ocasionales = (resumen.loc[s] for s in SEGMENTOS)
    verificar(leales["frecuencia_mensual"] == resumen["frecuencia_mensual"].max(), "Leales: deben ser los más frecuentes")
    verificar(leales["gasto_promedio"] == resumen["gasto_promedio"].max(), "Leales: deben tener el mayor gasto")
    verificar(madrugadores["pct_mañana"] >= 0.8, "Madrugadores: la mayoría debe preferir la mañana")
    verificar(riesgo["recencia_dias"] > max(leales["recencia_dias"], madrugadores["recencia_dias"]), "En riesgo: menos recientes que los activos")
    verificar(riesgo["frecuencia_mensual"] > ocasionales["frecuencia_mensual"], "En riesgo: compraban más que los Ocasionales")
    return resumen


def asignar_segmentos(pipeline, nombres_grupo, perfil_clientes):
    """Aplica un pipeline ya entrenado a perfiles nuevos o existentes."""
    grupos = pipeline.predict(perfil_clientes)
    return perfil_clientes.assign(segmento=pd.Series(grupos).map(nombres_grupo).to_numpy())


# --- Afinidad ------------------------------------------------------------------------------

def ventana_analisis(connection):
    """Primeros dos meses de compras: [inicio_mes1, fin_ventana)."""
    return connection.execute("""
        SELECT date_trunc('month', min(fecha_hora)) AS inicio_mes1,
               date_trunc('month', min(fecha_hora)) + INTERVAL 2 MONTH AS fin_ventana
        FROM compras
    """).fetchone()


def _segmentos_sql():
    return "[" + ", ".join(sql_text(segmento) for segmento in SEGMENTOS) + "]"


def calcular_afinidad(connection, perfil_clientes, inicio_mes1, fin_ventana):
    connection.register("segmentos_df", perfil_clientes[["cliente_id", "segmento"]])
    return connection.execute(f"""
        WITH compras_ventana AS (
            SELECT c.id AS compra_id, s.segmento
            FROM compras c
            JOIN segmentos_df s ON s.cliente_id = c.cliente_id
            WHERE c.fecha_hora >= {sql_text(inicio_mes1)}::TIMESTAMP
              AND c.fecha_hora < {sql_text(fin_ventana)}::TIMESTAMP
        ),
        compra_categoria AS (
            SELECT DISTINCT cv.compra_id, cv.segmento, p.categoria
            FROM compras_ventana cv
            JOIN lineas_compra l ON l.compra_id = cv.compra_id
            JOIN productos p ON p.id = l.producto_id
        ),
        compras_segmento AS (
            SELECT segmento, count(*)::DOUBLE AS compras FROM compras_ventana GROUP BY 1
        ),
        compras_total AS (
            SELECT count(*)::DOUBLE AS compras FROM compras_ventana
        ),
        categorias AS (
            SELECT DISTINCT categoria FROM productos
        ),
        categoria_segmento AS (
            SELECT segmento, categoria, count(*) AS compras FROM compra_categoria GROUP BY 1, 2
        ),
        categoria_total AS (
            SELECT categoria, count(*) AS compras FROM compra_categoria GROUP BY 1
        ),
        proporciones AS (
            SELECT
                cs.segmento,
                cat.categoria,
                coalesce(sc.compras, 0) / cs.compras AS prop_segmento,
                coalesce(ct.compras, 0) / t.compras AS prop_total
            FROM compras_segmento cs
            CROSS JOIN categorias cat
            CROSS JOIN compras_total t
            LEFT JOIN categoria_segmento sc ON sc.segmento = cs.segmento AND sc.categoria = cat.categoria
            LEFT JOIN categoria_total ct ON ct.categoria = cat.categoria
        )
        SELECT
            segmento,
            categoria,
            round(100 * prop_segmento, 2) AS pct_compras_segmento,
            round(100 * prop_total, 2) AS pct_compras_total,
            round(prop_segmento / prop_total, 4) AS indice_afinidad
        FROM proporciones
        ORDER BY list_position({_segmentos_sql()}, segmento), categoria
    """).df()


# --- Validación y exportación --------------------------------------------------------------

def validar_salidas(connection, perfil_clientes, afinidad, inicio_mes1, fin_ventana):
    num_clientes = connection.execute("SELECT count(*) FROM clientes").fetchone()[0]
    num_categorias = connection.execute("SELECT count(DISTINCT categoria) FROM productos").fetchone()[0]

    verificar(len(perfil_clientes) == num_clientes, "Cada cliente debe tener una fila en perfil_clientes")
    verificar(perfil_clientes["segmento"].isin(SEGMENTOS).all(), "Segmento fuera de los cuatro válidos")
    verificar(perfil_clientes["segmento"].nunique() == NUM_SEGMENTOS, "Los cuatro segmentos deben tener clientes")
    verificar(perfil_clientes.isna().sum().sum() == 0, "perfil_clientes no debe tener nulos")

    verificar(len(afinidad) == NUM_SEGMENTOS * num_categorias, "Una fila por segmento y categoría")
    verificar(not afinidad.duplicated(["segmento", "categoria"]).any(), "Combinaciones segmento-categoría repetidas")
    verificar(afinidad.notna().all().all(), "afinidad_segmento_categoria no debe tener nulos")
    verificar((afinidad["indice_afinidad"] >= 0).all(), "indice_afinidad no debe ser negativo")

    connection.register("segmentos_df", perfil_clientes[["cliente_id", "segmento"]])
    compras_por_segmento = connection.execute(f"""
        SELECT s.segmento, count(*) AS compras
        FROM compras c JOIN segmentos_df s ON s.cliente_id = c.cliente_id
        WHERE c.fecha_hora >= {sql_text(inicio_mes1)}::TIMESTAMP AND c.fecha_hora < {sql_text(fin_ventana)}::TIMESTAMP
        GROUP BY 1
    """).df().set_index("segmento")["compras"]
    peso_segmento = afinidad["segmento"].map(compras_por_segmento / compras_por_segmento.sum())
    consistencia = (
        afinidad
        .assign(ponderado=afinidad["pct_compras_segmento"] * peso_segmento)
        .groupby("categoria")
        .agg(ponderado=("ponderado", "sum"), total=("pct_compras_total", "first"))
    )
    verificar(np.allclose(consistencia["ponderado"], consistencia["total"], atol=0.05), "pct_compras_total inconsistente con los segmentos")
    return {"perfil_clientes": num_clientes, "afinidad_segmento_categoria": NUM_SEGMENTOS * num_categorias}


def exportar_gold(connection, perfil_clientes, afinidad, gold, filas_esperadas):
    gold = Path(gold)
    gold.mkdir(parents=True, exist_ok=True)
    connection.register("perfil_clientes_df", perfil_clientes[COLUMNAS_PERFIL])
    connection.register("afinidad_df", afinidad)
    connection.execute(f"COPY (SELECT * FROM perfil_clientes_df ORDER BY cliente_id) TO {sql_text(gold / 'perfil_clientes.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)")
    connection.execute(f"COPY (SELECT * FROM afinidad_df ORDER BY list_position({_segmentos_sql()}, segmento), categoria) TO {sql_text(gold / 'afinidad_segmento_categoria.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)")

    for name, esperado in filas_esperadas.items():
        filas = connection.execute(f"SELECT count(*) FROM read_parquet({sql_text(gold / f'{name}.parquet')})").fetchone()[0]
        verificar(filas == esperado, f"{name}: se exportaron {filas} filas, se esperaban {esperado}")
    sin_segmento = connection.execute(
        f"SELECT count(*) FROM read_parquet({sql_text(gold / 'perfil_clientes.parquet')}) WHERE segmento IS NULL"
    ).fetchone()[0]
    verificar(sin_segmento == 0, "Hay clientes sin segmento en el Parquet exportado")
