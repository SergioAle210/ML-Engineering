# Taller 3: MLflow con scikit-learn

Machine Learning Engineering (CC3105). Taller para conocer MLflow e integrarlo con un pipeline de scikit-learn.

Repositorio: https://github.com/SergioAle210/ML-Engineering/tree/main/Talleres/Taller3

## Contenido de la carpeta

| Archivo | Contenido | Sección de este documento |
|---|---|---|
| entrega.md | Este documento | Todas |
| notebooks/02_tutorial_jupyter_mlflow.ipynb | Tutorial en Jupyter y datos sintéticos | Inciso 2 |
| notebooks/03_pipeline_sklearn_mlflow.ipynb | MLflow con un pipeline de scikit-learn | Inciso 3 |
| capturas/ | Capturas de la interfaz de MLflow | Capturas |
| requirements.txt | Paquetes usados | Cómo reproducir |

## Qué es MLflow

MLflow guarda lo que pasa cuando se entrena un modelo. Registra parámetros, métricas y modelos de cada intento. Una interfaz web permite compararlos.

Términos que se usan en este documento:

1. **Experimento:** agrupa los intentos de un mismo problema.
2. **Run:** un intento de entrenamiento dentro de un experimento.
3. **Parámetro:** un valor elegido antes de entrenar.
4. **Métrica:** un resultado medido, como el error.
5. **Artefacto:** un archivo guardado en el run, como el modelo.

## Cómo reproducir

1. Crear y activar un ambiente virtual en esta carpeta.
2. Instalar los paquetes con `pip install -r requirements.txt`.
3. Iniciar el servidor con `mlflow server --port 5000`.
4. Abrir http://localhost:5000 en el navegador.

## Inciso 1: Tres páginas de la documentación

### Página 1: Tracking Quickstart

Enlace: https://mlflow.org/docs/latest/ml/tracking/quickstart/

La elegimos por estas razones:

1. Reúne en un solo lugar los pasos básicos que se repiten en todo el taller.
2. Muestra el registro automático con `autolog` y el registro manual de parámetros y métricas.
3. Enseña a ver el run en la interfaz y a cargar el modelo para predecir.
4. Usa el dataset Iris, que es pequeño y fácil de seguir.

### Página 2: Guía de scikit-learn con MLflow

Enlace: https://www.mlflow.org/docs/3.5.0/ml/traditional-ml/sklearn/guide/

La elegimos por estas razones:

1. Responde directo al inciso 3. Su sección Advanced Features tiene una pestaña Pipeline Integration.
2. Explica que `autolog` registra los parámetros de cada paso del pipeline.
3. Explica cómo MLflow maneja la búsqueda de hiperparámetros. Ya hicimos una en la Actividad 1.

Usamos la versión 3.5.0 porque en la versión más reciente de esta guía no encontramos la sección de pipelines.

### Página 3: Model Registry

Enlace: https://mlflow.org/docs/latest/ml/model-registry/

La elegimos por estas razones:

1. Muestra qué pasa después de entrenar. El modelo se registra con un nombre y sus versiones.
2. Un alias como `champion` apunta a una versión. Se puede cambiar de versión sin tocar el código que usa el modelo.
3. Ayuda a pensar usos reales de MLflow.

## Inciso 2: Tutorial de MLflow en Jupyter

Tutorial seguido: MLflow with Optuna, Hyperparameter Optimization and Tracking.

Enlace: https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/hyperparameter-tuning-with-child-runs/

Lo elegimos porque incluye la sección Generate our synthetic training data. El trabajo está en `notebooks/02_tutorial_jupyter_mlflow.ipynb`.

> Por completar: pasos realizados y número de pruebas de Optuna usado (el tutorial original usa 500).

### Inciso 2a: Análisis de los datos sintéticos

Cómo se generan los datos según el tutorial:

1. La función `generate_apple_sales_data_with_promo_adjustment` crea 5000 filas por defecto.
2. Usa una semilla fija: `np.random.seed(9999)`.
3. El ruido sale de una distribución normal con media 0 y desviación 50.
4. Las variables son temperatura, lluvia, fin de semana, feriado, precio, promoción, demanda del día anterior, precio de la competencia e intensidad de marketing.
5. Los datos se dividen en 75 por ciento para entrenar y 25 por ciento para validar. No se ve una semilla en ese paso.
6. El modelo es XGBoost y Optuna busca sus parámetros.

Puntos a evaluar para mejorar el algoritmo:

1. Efecto de cambiar la semilla en los resultados.
2. Fijar `random_state` en la división de datos para poder repetirla.
3. Mezclar las filas al dividir, cuando existe una columna de fecha y una de demanda del día anterior.
4. Si un ruido fijo de 50 se parece al de un negocio real.
5. Comparar XGBoost contra un modelo base simple, como una regresión lineal.
6. Cuántas pruebas de Optuna hacen falta para ver una mejora real.

> Por completar: mejoras probadas y sus resultados. Cada cambio se registra como un run nuevo en MLflow.

## Inciso 3: MLflow con un pipeline de scikit-learn

La sección de la documentación está en la página 2: Advanced Features, pestaña Pipeline Integration.

La documentación indica que al activar `mlflow.sklearn.autolog()` antes de entrenar, MLflow registra los parámetros de cada paso del pipeline. También captura las métricas de `score`.

Pipeline usado: el de churn de la Actividad 1, copiado en `Talleres/Taller2/pipeline`. El trabajo está en `notebooks/03_pipeline_sklearn_mlflow.ipynb`.

> Por completar: descripción de lo implementado y de lo que se observa en la interfaz.

## Capturas de MLflow

> Por completar: capturas guardadas en `capturas/`. Se sugiere la lista de runs, la gráfica de comparación, la página de un run con sus parámetros y métricas, y el run del pipeline del inciso 3.

## Conclusiones

> Por completar.

<!--
Preguntas de apoyo para las conclusiones:
1. Qué problema de nuestros proyectos resolvería MLflow.
2. Qué parte usaríamos primero: el registro de runs o el Model Registry.
3. Qué dificultades tuvimos al usarlo.
4. Si lo usaríamos en un proyecto personal o profesional y por qué.
-->

## Fuentes

1. https://mlflow.org/docs/latest/ml/tracking/quickstart/
2. https://www.mlflow.org/docs/3.5.0/ml/traditional-ml/sklearn/guide/
3. https://mlflow.org/docs/latest/ml/model-registry/
4. https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/hyperparameter-tuning-with-child-runs/
