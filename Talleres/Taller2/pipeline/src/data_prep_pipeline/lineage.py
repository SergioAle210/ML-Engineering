"""Data lineage: where every row and every column in the pipeline comes from.

CRISP-DM mapping: Data Understanding / Data Preparation — being able to answer
"which raw field produced this model feature, and which rows were dropped on
the way?" without reading the source of every stage.

The lineage is *derived from the pipeline objects themselves*, not written down
by hand: `trace_stages` runs the real stages and records the shape at each
boundary, and `trace_columns` reads the fitted `ColumnTransformer` to map each
source column to the features it produced. That way the report cannot silently
drift out of date when a stage changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.pipeline import Pipeline

from .extraction import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from .filtering import MAX_PLAUSIBLE_TENURE, MIN_PLAUSIBLE_TENURE
from .pipeline import build_cleaning_pipeline, build_preprocessing_pipeline, split_dataset

PREPROCESSING_STEP = "preprocessing"


def _outputs_per_column(transformer, columns: list[str]) -> list[int]:
    """How many output features each input column of `transformer` produces.

    One-hot encoding expands a column into one feature per learned category;
    every other transformer used here maps one column to one column. Reading
    `categories_` off the fitted encoder avoids parsing feature-name strings,
    which would be ambiguous whenever a category value contains an underscore
    (e.g. `PaymentMethod_Bank transfer (automatic)`).
    """
    last_step = transformer.steps[-1][1] if isinstance(transformer, Pipeline) else transformer
    categories = getattr(last_step, "categories_", None)
    if categories is None:
        return [1] * len(columns)
    drop_idx = getattr(last_step, "drop_idx_", None)
    return [
        len(cats) - (1 if drop_idx is not None and drop_idx[i] is not None else 0)
        for i, cats in enumerate(categories)
    ]


def trace_columns(fitted_preprocessing: Pipeline) -> pd.DataFrame:
    """Maps every source column to the model features it produced.

    Takes the preprocessing Pipeline *after* it has been fit, since the number
    of one-hot features is only known once the encoder has seen the data.
    """
    column_transformer = fitted_preprocessing.named_steps[PREPROCESSING_STEP]
    all_features = list(column_transformer.get_feature_names_out())

    rows: list[dict] = []
    offset = 0
    for block_name, transformer, columns in column_transformer.transformers_:
        if transformer == "drop" or not len(columns):
            continue
        steps = [type(step).__name__ for _, step in transformer.steps]
        for column, n_out in zip(columns, _outputs_per_column(transformer, columns)):
            produced = all_features[offset : offset + n_out]
            offset += n_out
            rows.append(
                {
                    "source_column": column,
                    "block": block_name,
                    "transform": " -> ".join(steps),
                    "n_features_out": n_out,
                    "features_out": ", ".join(produced),
                }
            )
    return pd.DataFrame(rows)


@dataclass
class LineageReport:
    """Everything `build_lineage_report` measured, ready to display or print."""

    stages: pd.DataFrame
    columns: pd.DataFrame
    summary: dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = ["LINAJE DE DATOS — resumen"]
        for key, value in self.summary.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


def build_lineage_report(
    raw_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> LineageReport:
    """Runs the real pipeline stages and records what each one did.

    Returns stage-level lineage (rows/columns in and out of every stage) and
    column-level lineage (source column -> model features).
    """
    raw_rows, raw_cols = raw_df.shape

    # --- extraction ------------------------------------------------------
    cleaning_pipeline = build_cleaning_pipeline()
    extractor = cleaning_pipeline.named_steps["extraction"]
    extracted = extractor.fit_transform(raw_df)
    dropped_at_extraction = [c for c in raw_df.columns if c not in extracted.columns]
    # Blanks in the text TotalCharges column become NaN here (they are new
    # customers with tenure 0) and are median-imputed later, in preprocessing —
    # a provenance fact worth surfacing: those values are derived, not observed.
    coerced_to_null = int(extracted["TotalCharges"].isna().sum())

    # --- filtering -------------------------------------------------------
    duplicate_ids = int(extracted[ID_COLUMN].duplicated().sum())
    deduplicated = extracted.drop_duplicates(subset=[ID_COLUMN])
    out_of_range = int(
        (~deduplicated["tenure"].between(MIN_PLAUSIBLE_TENURE, MAX_PLAUSIBLE_TENURE)).sum()
    )
    clean_df = cleaning_pipeline.named_steps["filtering"].fit_transform(deduplicated)

    # --- split -----------------------------------------------------------
    X_train, X_test, y_train, y_test = split_dataset(
        clean_df, test_size=test_size, random_state=random_state
    )

    # --- preprocessing ---------------------------------------------------
    preprocessing = build_preprocessing_pipeline()
    X_train_out = preprocessing.fit_transform(X_train)
    n_features_out = X_train_out.shape[1]

    stages = pd.DataFrame(
        [
            {
                "orden": 1,
                "etapa": "Fuente",
                "modulo": "data_prep_pipeline.data_source",
                "filas_entrada": None,
                "filas_salida": raw_rows,
                "columnas_entrada": None,
                "columnas_salida": raw_cols,
                "detalle": "CSV Telco Customer Churn leido desde ruta local, Volume o URL publica",
            },
            {
                "orden": 2,
                "etapa": "Extraccion",
                "modulo": "extraction.DataExtractor",
                "filas_entrada": raw_rows,
                "filas_salida": len(extracted),
                "columnas_entrada": raw_cols,
                "columnas_salida": extracted.shape[1],
                "detalle": (
                    f"Selecciona {len(FEATURE_COLUMNS)} features + {ID_COLUMN} + {TARGET_COLUMN}; "
                    f"convierte TotalCharges de texto a numerico "
                    f"({coerced_to_null} valores en blanco -> nulo, imputados luego con la mediana); "
                    f"descarta columnas: {dropped_at_extraction or 'ninguna'}"
                ),
            },
            {
                "orden": 3,
                "etapa": "Filtrado",
                "modulo": "filtering.DataFilter",
                "filas_entrada": len(extracted),
                "filas_salida": len(clean_df),
                "columnas_entrada": extracted.shape[1],
                "columnas_salida": clean_df.shape[1],
                "detalle": (
                    f"Elimina {duplicate_ids} {ID_COLUMN} duplicados y {out_of_range} filas con "
                    f"tenure fuera de [{MIN_PLAUSIBLE_TENURE}, {MAX_PLAUSIBLE_TENURE}]; "
                    f"descarta la columna {ID_COLUMN} tras usarla"
                ),
            },
            {
                "orden": 4,
                "etapa": "Split",
                "modulo": "pipeline.split_dataset",
                "filas_entrada": len(clean_df),
                "filas_salida": len(X_train),
                "columnas_entrada": clean_df.shape[1],
                "columnas_salida": X_train.shape[1],
                "detalle": (
                    f"Separa X/y y hace split estratificado {1 - test_size:.0%}/{test_size:.0%} "
                    f"(train={len(X_train)}, test={len(X_test)}); "
                    f"{TARGET_COLUMN} Yes/No -> 1/0"
                ),
            },
            {
                "orden": 5,
                "etapa": "Preprocesamiento",
                "modulo": "pipeline.build_preprocessing_pipeline",
                "filas_entrada": len(X_train),
                "filas_salida": X_train_out.shape[0],
                "columnas_entrada": X_train.shape[1],
                "columnas_salida": n_features_out,
                "detalle": (
                    f"ColumnTransformer ajustado SOLO con X_train: {len(NUMERIC_FEATURES)} numericas "
                    f"(mediana + escalado), {len(CATEGORICAL_FEATURES)} categoricas (moda + one-hot)"
                ),
            },
            {
                "orden": 6,
                "etapa": "Modelo",
                "modulo": "model.build_model_pipeline",
                "filas_entrada": len(X_train),
                "filas_salida": None,
                "columnas_entrada": n_features_out,
                "columnas_salida": 1,
                "detalle": "GradientBoostingClassifier -> probabilidad de churn por cliente",
            },
        ]
    )

    columns = trace_columns(preprocessing)

    summary = {
        "filas crudas": raw_rows,
        "filas tras limpieza": len(clean_df),
        "filas descartadas": raw_rows - len(clean_df),
        "columnas crudas": raw_cols,
        "columnas fuente usadas": len(FEATURE_COLUMNS),
        "features del modelo": n_features_out,
        "expansion one-hot": f"{len(CATEGORICAL_FEATURES)} categoricas -> "
        f"{n_features_out - len(NUMERIC_FEATURES)} features",
        "objetivo": f"{TARGET_COLUMN} (Yes/No -> 1/0)",
        "valores imputados (TotalCharges)": coerced_to_null,
    }
    return LineageReport(stages=stages, columns=columns, summary=summary)
