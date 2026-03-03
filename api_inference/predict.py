import pandas as pd
#from train import load_model
import pickle


EXPECTED_NUM_COLS = [
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]
"""
Lista de nombres de columnas numéricas esperadas en los datos de entrada.
"""

EXPECTED_CAT_COLS = [
    "island",
    "sex"
]
"""
Lista de nombres de columnas categóricas esperadas en los datos de entrada.
"""

ALL_EXPECTED_COLS = EXPECTED_NUM_COLS + EXPECTED_CAT_COLS
"""
Lista de todas las columnas esperadas (numéricas + categóricas).
"""

def load_model(filename):
    """Carga un modelo entrenado junto con sus componentes de preprocesamiento.
    
    Deserializa el modelo, los encoders y el scaler desde un archivo pickle
    previamente guardado.
    
    Args:
        filename (str): Ruta y nombre del archivo desde donde cargar el modelo.
        
    Returns:
        tuple: Tupla contiendo:
            - model: Modelo entrenado (DecisionTreeClassifier, SVC o KNeighborsClassifier).
            - encoders (dict): Diccionario con los OneHotEncoders para variables categóricas.
            - scaler: StandardScaler utilizado para normalizar características (puede ser None).
            
    Raises:
        FileNotFoundError: Si el archivo especificado no existe.
        pickle.UnpicklingError: Si el archivo no contiene un objeto pickle válido.
        
    Note:
        - El archivo debe haber sido creado con la función save_model()
        - Los componentes devueltos se pueden usar para realizar predicciones en nuevos datos
    
    Example:
        >>> model, encoders, scaler = load_model('models/my_model.pkl')
    """
    with open(filename, "rb") as f:
        data = pickle.load(f)

    return data["model"], data["encoders"], data["scaler"]


def predict_new_data(df_new, model, encoders, scaler=None):
    """
    Realiza la predicción de la especie de pingüino para nuevos datos usando un modelo entrenado.

    Preprocesa los datos de entrada de la misma forma que en el entrenamiento:
    - Rellena valores faltantes en columnas numéricas con la mediana
    - Rellena valores faltantes en columnas categóricas con "Unknown"
    - Aplica OneHotEncoder a las variables categóricas
    - Aplica escalado si se proporciona un scaler
    - Asegura el mismo orden y cantidad de columnas que durante el entrenamiento

    Args:
        df_new (pd.DataFrame): DataFrame con los nuevos datos a predecir.
        model: Modelo entrenado (DecisionTreeClassifier, SVC, KNeighborsClassifier, etc.).
        encoders (dict): Diccionario con el OneHotEncoder bajo la clave 'onehot'.
        scaler (StandardScaler, opcional): Scaler entrenado, si se usó durante el entrenamiento.

    Returns:
        np.ndarray: Array con las predicciones del modelo para los datos de entrada.
    """
    df_new = df_new.copy()
    num_cols = [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]
    cat_cols = ["island", "sex"]

    # Limpieza numéricas
    df_new[num_cols] = df_new[num_cols].fillna(df_new[num_cols].median())
    # Limpieza categóricas
    df_new["sex"] = df_new["sex"].fillna("Unknown")

    # OneHotEncoder para variables categóricas
    ohe = encoders["onehot"]
    X_cat = ohe.transform(df_new[cat_cols])
    cat_feature_names = ohe.get_feature_names_out(cat_cols)
    X_cat_df = pd.DataFrame(X_cat, columns=cat_feature_names, index=df_new.index)

    # Concatenar numéricas y categóricas
    X_final = pd.concat([df_new[num_cols], X_cat_df], axis=1)

    # Asegurar el orden de columnas igual al entrenamiento
    if hasattr(ohe, 'feature_names_in_'):
        ordered_cols = list(num_cols) + list(ohe.get_feature_names_out(cat_cols))
        X_final = X_final[ordered_cols]

    # Escalado opcional
    if scaler is not None:
        X_scaled = scaler.transform(X_final)
        X_final = pd.DataFrame(X_scaled, columns=X_final.columns, index=X_final.index)

    # Predicción
    return model.predict(X_final)
