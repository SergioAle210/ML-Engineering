"""Segmentación de clientes del caso de estudio 2 (K-means sobre perfil_clientes)."""

from .pipeline import (
    SEGMENTOS,
    construir_pipeline,
    construir_preprocesamiento,
    entrenar,
    nombrar_grupos,
)

__version__ = "0.1.0"
