"""
Penguin Prediction API
----------------------

Servicio REST construido con FastAPI para predecir la especie de un pingüino
a partir de sus características físicas.

Características:
- Permite ejecutar múltiples modelos en una sola petición
- Modelos cargados en memoria al iniciar la aplicación
- Validación automática de parámetros mediante Query + Enum
- Preparado para despliegue en Docker

Autor: Taller MLOps
Versión: 1.0.0
"""

from fastapi import FastAPI, Query, Request,  Response
import pandas as pd
from typing import List, Annotated
from predict import predict_new_data, load_model
from enum import Enum
import logging
from logging.handlers import RotatingFileHandler
import os
import time 
from fastapi.responses import JSONResponse





# ------------------------------------------------------------------------------
# Inicialización de la aplicación
# ------------------------------------------------------------------------------

app = FastAPI(
    title="Taller 1 MLOPS: Penguin Prediction API",
    description="Servicio para clasificar pingüinos según características físicas",
    version="1.0.0",
)


logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Carga de modelos
# ------------------------------------------------------------------------------
# Los modelos se cargan una única vez al iniciar el contenedor.
# Esto evita overhead de I/O y deserialización en cada request (muy importante en producción).

tree_model, tree_encoders, tree_scaler = load_model("decision_tree.pkl")
knn_model, knn_encoders, knn_scaler = load_model("knn.pkl")
svm_model, svm_encoders, svm_scaler = load_model("svm.pkl")

# Registro centralizado de modelos disponibles
# Permite seleccionar dinámicamente qué modelo ejecutar desde la API
models_dict = {
    "TREE": {"model": tree_model, "encoders": tree_encoders, "scaler": tree_scaler},
    "KNN": {"model": knn_model, "encoders": knn_encoders, "scaler": knn_scaler},
    "SVM": {"model": svm_model, "encoders": svm_encoders, "scaler": svm_scaler},
}


# ------------------------------------------------------------------------------
# Enumeraciones (validación fuerte de parámetros de entrada)
# ------------------------------------------------------------------------------

class islas_class(str, Enum):
    """Isla de procedencia del pingüino."""
    Torgersen = "Torgersen"
    Dream = "Dream"
    Biscoe = "Biscoe"


class sex_class(str, Enum):
    """Sexo biológico del pingüino."""
    Male = "Male"
    Female = "Female"


class model_class(str, Enum):
    """Modelos de ML disponibles en el servicio."""
    TREE = "TREE"
    KNN = "KNN"
    SVM = "SVM"



# ------------------------------------------------------------------------------
# Endpoint principal
# ------------------------------------------------------------------------------

@app.post(
    "/predict",
    summary="Predecir especie",
    description="Predice la especie del pingüino utilizando uno o varios modelos de ML",
)
async def root(
    models: Annotated[
        List[model_class],
        Query(..., description="Modelos a utilizar: TREE, KNN, SVM. Para seleccionar más de un modelo, haga Ctrl + click sobre los modelos deseados."),
    ],
    culmen_length_mm: float = Query(39, description="Longitud del culmen en mm"),
    culmen_depth_mm: float = Query(18.7, description="Profundidad del culmen en mm"),
    flipper_length_mm: float = Query(180, description="Longitud de la aleta en mm"),
    body_mass_g: float = Query(3700, description="Masa corporal en gramos"),
    island: islas_class = Query(islas_class.Torgersen, description="Isla de origen"),
    sex: sex_class = Query(sex_class.Male, description="Sexo del ejemplar"),
):
    """
    Realiza la predicción de especie para un pingüino.

    El endpoint permite ejecutar múltiples modelos en paralelo lógico
    (misma entrada, múltiples inferencias) y devuelve un diccionario
    con cada modelo como clave.

    Returns
    -------
    dict
        {
            "TREE": ["Adelie"],
            "SVM": ["Adelie"]
        }
    """

    # --------------------------------------------------------------------------
    # Construcción del DataFrame de entrada
    # Los modelos esperan un DataFrame con exactamente el mismo esquema
    # usado durante el entrenamiento.
    # --------------------------------------------------------------------------
    df = pd.DataFrame([{
        "culmen_length_mm": culmen_length_mm,
        "culmen_depth_mm": culmen_depth_mm,
        "flipper_length_mm": flipper_length_mm,
        "body_mass_g": body_mass_g,
        "island": island,
        "sex": sex
    }])

    logger.info(f"input recibido: {df.to_json()}")

    response = {}

    # --------------------------------------------------------------------------
    # Ejecución de inferencia por cada modelo solicitado
    # --------------------------------------------------------------------------
    for m in models:
        model_name = m.value

        prediction = predict_new_data(
            df,
            models_dict[model_name]["model"],
            models_dict[model_name]["encoders"],
            models_dict[model_name]["scaler"]
        )

        # Conversión a lista para serialización JSON
        response[model_name] = prediction.tolist()
    logger.info(f"Response enviado: {response}")

    return response
