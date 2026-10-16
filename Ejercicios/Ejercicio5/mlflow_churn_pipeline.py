"""Pipeline de churn (scikit-learn) integrado con MLflow: Experiments y Models.

Toma el pipeline de la Actividad 1 y lo conecta con MLflow.
Funciona en dos modos sin cambiar el codigo:
1. Databricks: usa el tracking del workspace y registra el modelo en Unity Catalog.
2. Local: guarda todo en mlflow.db (SQLite) para poder probarlo.

Uso en Databricks:
    python mlflow_churn_pipeline.py --catalog workspace --schema default

Uso local:
    python mlflow_churn_pipeline.py
"""

import argparse
import os

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
MODEL_ALIAS = "champion"

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MIN_PLAUSIBLE_TENURE = 0
MAX_PLAUSIBLE_TENURE = 100


class DataExtractor(BaseEstimator, TransformerMixin):
    """Selecciona las columnas relevantes y corrige el tipo de TotalCharges."""

    def __init__(self, columns=None, include_target=True):
        self.columns = columns
        self.include_target = include_target

    def fit(self, X, y=None):
        base_columns = self.columns or FEATURE_COLUMNS
        self._columns = [ID_COLUMN] + base_columns
        if self.include_target and TARGET_COLUMN not in self._columns:
            self._columns = self._columns + [TARGET_COLUMN]
        return self

    def transform(self, X):
        missing = [c for c in self._columns if c not in X.columns]
        if missing:
            raise KeyError(f"Faltan columnas esperadas en el origen: {missing}")
        result = X[self._columns].copy()
        result["TotalCharges"] = pd.to_numeric(result["TotalCharges"], errors="coerce")
        return result


class DataFilter(BaseEstimator, TransformerMixin):
    """Elimina clientes duplicados y filas con valores poco creibles."""

    def __init__(self, min_tenure=MIN_PLAUSIBLE_TENURE, max_tenure=MAX_PLAUSIBLE_TENURE):
        self.min_tenure = min_tenure
        self.max_tenure = max_tenure

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.drop_duplicates(subset=[ID_COLUMN])
        tenure_en_rango = result["tenure"].between(self.min_tenure, self.max_tenure)
        result = result.loc[tenure_en_rango]
        return result.drop(columns=[ID_COLUMN]).copy()


def build_cleaning_pipeline():
    return Pipeline(steps=[("extraction", DataExtractor()), ("filtering", DataFilter())])


def build_preprocessing_pipeline():
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    return Pipeline(steps=[("preprocessing", preprocessor)])


def split_dataset(clean_df, test_size=0.2, random_state=42):
    X = clean_df[FEATURE_COLUMNS]
    y = clean_df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# Hiperparametros ganadores de la busqueda hecha en la Actividad 1.
MODEL_PARAMS = {"learning_rate": 0.1, "max_depth": 2, "n_estimators": 100, "random_state": 42}


def build_model_pipeline():
    return Pipeline(steps=[
        ("preprocessing", build_preprocessing_pipeline()),
        ("classifier", GradientBoostingClassifier(**MODEL_PARAMS)),
    ])


def is_databricks():
    """Detecta si el codigo corre dentro de Databricks."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def configure_mlflow(args):
    """Configura tracking, registry y experimento. Devuelve el nombre del modelo."""
    if is_databricks():
        mlflow.set_registry_uri("databricks-uc")
        if args.experiment:
            mlflow.set_experiment(args.experiment)
        return f"{args.catalog}.{args.schema}.{args.model_name}"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(args.experiment or "mlflow-churn-pipeline")
    return args.model_name


def prepare_data(data_path):
    """Lee el CSV, limpia los datos y los divide en entrenamiento y prueba."""
    raw_df = pd.read_csv(data_path or SOURCE_URL)
    clean_df = build_cleaning_pipeline().fit_transform(raw_df)
    return split_dataset(clean_df)


def train_and_log(model_name, X_train, X_test, y_train, y_test):
    """Entrena el pipeline, lo registra en MLflow y devuelve el run_id."""
    # Autolog guarda parametros y metricas. El modelo se registra a mano con firma.
    mlflow.sklearn.autolog(log_models=False)

    with mlflow.start_run(run_name="churn-gradient-boosting") as run:
        model = build_model_pipeline()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        mlflow.log_metrics(
            {
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_precision": precision_score(y_test, y_pred),
                "test_recall": recall_score(y_test, y_pred),
                "test_f1": f1_score(y_test, y_pred),
                "test_roc_auc": roc_auc_score(y_test, y_proba),
            }
        )
        mlflow.set_tags({"dataset": "telco_customer_churn", "etapa_crisp_dm": "modeling"})

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            # El formato por defecto de MLflow rechaza los arboles de Gradient Boosting.
            serialization_format="cloudpickle",
            registered_model_name=model_name,
        )
        print(f"Run ID: {run.info.run_id}")
        return run.info.run_id


def promote_latest_version(model_name):
    """Asigna el alias champion a la version mas reciente del modelo."""
    client = MlflowClient()
    versions = client.search_model_versions(f"name = '{model_name}'")
    latest = max(versions, key=lambda version: int(version.version))
    client.set_registered_model_alias(model_name, MODEL_ALIAS, latest.version)
    print(f"El alias {MODEL_ALIAS} apunta a la version {latest.version}")


def predict_with_alias(model_name, X_test, y_test):
    """Carga el modelo por alias y predice, como lo haria un consumidor."""
    loaded = mlflow.pyfunc.load_model(f"models:/{model_name}@{MODEL_ALIAS}")
    sample = X_test.head(5)
    print(pd.DataFrame({"real": y_test.head(5).values, "predicho": loaded.predict(sample)}))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="workspace", help="Catalogo de Unity Catalog")
    parser.add_argument("--schema", default="default", help="Esquema de Unity Catalog")
    parser.add_argument("--model-name", default="churn_gradient_boosting")
    parser.add_argument("--experiment", default=None, help="Ruta del experimento")
    parser.add_argument("--data-path", default=None, help="CSV local en vez de la URL publica")
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = configure_mlflow(args)
    X_train, X_test, y_train, y_test = prepare_data(args.data_path)
    train_and_log(model_name, X_train, X_test, y_train, y_test)
    promote_latest_version(model_name)
    predict_with_alias(model_name, X_test, y_test)


if __name__ == "__main__":
    main()
