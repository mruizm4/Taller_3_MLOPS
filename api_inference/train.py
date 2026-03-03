import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


def preprocess_data(df):
    """Preprocesa los datos del conjunto de datos de pingüinos usando OneHotEncoder para variables categóricas.
    
    Realiza las siguientes operaciones:
    - Rellena valores faltantes en columnas numéricas con la mediana
    - Rellena valores faltantes en columnas categóricas con "Unknown"
    - Codifica las variables categóricas utilizando OneHotEncoder
    - Separa las características (X) del target (y)
    
    Args:
        df (pd.DataFrame): DataFrame con los datos crudos de pingüinos.
        
    Returns:
        tuple: Tupla contiendo:
            - X (pd.DataFrame): Características preprocesadas
            - y (pd.Series): Variable objetivo (especies)
            - encoders (dict): Diccionario con los OneHotEncoders para cada columna categórica
    
    Example:
        >>> df = pd.read_csv('penguins.csv')
        >>> X, y, encoders = preprocess_data(df)
    """
    df = df.copy()

    num_cols = [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]

    cat_cols = ["island", "sex"]
    target = "species"

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df["sex"] = df["sex"].fillna("Unknown")

    X = df.drop(columns=target)
    y = df[target]

    # OneHotEncoder para variables categóricas
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = ohe.fit_transform(X[cat_cols])
    cat_feature_names = ohe.get_feature_names_out(cat_cols)
    X_cat_df = pd.DataFrame(X_cat, columns=cat_feature_names, index=X.index)

    # Concatenar numéricas y categóricas
    X_final = pd.concat([X[num_cols], X_cat_df], axis=1)

    encoders = {"onehot": ohe}

    return X_final, y, encoders



def train_decision_tree(df):
    """Entrena un modelo de Árbol de Decisión usando OneHotEncoder.
    
    Preprocesa los datos, divide en conjuntos de entrenamiento y prueba,
    entrena un DecisionTreeClassifier y evalúa su desempeño.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos crudos de pingüinos.
        
    Returns:
        DecisionTreeClassifier: Modelo entrenado de Árbol de Decisión.
        
    Note:
        - Utiliza max_depth=5 para evitar overfitting
        - Aplica class_weight='balanced' para manejar desbalance de clases
        - Guarda los resultados en 'models_performance/decision_tree_results.txt'
        - Guarda el modelo en 'models/decision_tree.pkl'
    """
    X, y, encoders = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    return evaluate_model(
        model, X_test, y_test,
        encoders,
        scaler=None,
        model_name="decision_tree"
    )


def train_svm(df):
    """Entrena un modelo de SVM usando OneHotEncoder.
    
    Preprocesa los datos, aplica normalización estándar, divide en conjuntos
    de entrenamiento y prueba, entrena un SVC y evalúa su desempeño.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos crudos de pingüinos.
        
    Returns:
        SVC: Modelo entrenado de SVM.
        
    Note:
        - Utiliza kernel='rbf' para capturar relaciones no lineales
        - Aplica StandardScaler para normalizar las características
        - Aplica class_weight='balanced' para manejar desbalance de clases
        - Guarda los resultados en 'models_performance/svm_results.txt'
        - Guarda el modelo y el scaler en 'models/svm.pkl'
    """
    X, y, encoders = preprocess_data(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=42
    )

    model.fit(X_train, y_train)

    return evaluate_model(
        model, X_test, y_test,
        encoders,
        scaler=scaler,
        model_name="svm"
    )



def train_knn(df):
    """Entrena un modelo de KNN usando OneHotEncoder.
    
    Preprocesa los datos, aplica normalización estándar, divide en conjuntos
    de entrenamiento y prueba, entrena un KNeighborsClassifier y evalúa su desempeño.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos crudos de pingüinos.
        
    Returns:
        KNeighborsClassifier: Modelo entrenado de KNN.
        
    Note:
        - Utiliza n_neighbors=5 para determinar la clasificación
        - Aplica StandardScaler para normalizar las características
        - Guarda los resultados en 'models_performance/knn_results.txt'
        - Guarda el modelo y el scaler en 'models/knn.pkl'
    """
    X, y, encoders = preprocess_data(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = KNeighborsClassifier(n_neighbors=5)

    model.fit(X_train, y_train)

    return evaluate_model(
        model, X_test, y_test,
        encoders,
        scaler=scaler,
        model_name="knn"
    )


def evaluate_model(model, X_test, y_test, encoders, scaler, model_name):
    """Evalúa el desempeño de un modelo entrenado.
    
    Genera predicciones, calcula métricas de evaluación (reporte de clasificación
    y matriz de confusión), guarda los resultados en un archivo de texto y
    persiste el modelo junto con sus componentes.
    
    Args:
        model: Modelo entrenado (DecisionTreeClassifier, SVC o KNeighborsClassifier).
        X_test (array-like): Características del conjunto de prueba.
        y_test (array-like): Etiquetas verdaderas del conjunto de prueba.
        encoders (dict): Diccionario con los OneHotEncoders para variables categóricas.
        scaler: StandardScaler utilizado para normalizar X (None si no se usó).
        model_name (str): Nombre del modelo (ej: 'decision_tree', 'svm', 'knn').
        
    Returns:
        model: El mismo modelo pasado como argumento.
        
    Note:
        - Genera un reporte de clasificación con precisión, recall y f1-score
        - Genera una matriz de confusión
        - Guarda los resultados en 'models_performance/{model_name}_results.txt'
        - Persiste el modelo en 'models/{model_name}.pkl'
    """
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    results_text = f"""
MODEL: {model_name.upper()}

CLASSIFICATION REPORT
{report}

CONFUSION MATRIX
{matrix}
"""

    with open(f"models_performance/{model_name}_results.txt", "w") as f:
        f.write(results_text)

    save_model(model, encoders, scaler, f"models/{model_name}.pkl")

    return model


def save_model(model, encoders, scaler, filename):
    """Guarda un modelo entrenado junto con sus componentes de preprocesamiento.
    
    Serializa el modelo, los encoders y el scaler en un archivo pickle para
    posterior utilización en predicciones.
    
    Args:
        model: Modelo entrenado (DecisionTreeClassifier, SVC o KNeighborsClassifier).
        encoders (dict): Diccionario con los OneHotEncoders para variables categóricas.
        scaler: StandardScaler utilizado para normalizar características (puede ser None).
        filename (str): Ruta y nombre del archivo donde guardar el modelo.
        
    Returns:
        None
        
    Note:
        - Guarda un diccionario con tres claves: 'model', 'encoders' y 'scaler'
        - Utiliza el protocolo binario de pickle
        - El archivo se crea o se sobrescribe si ya existe
    
    Example:
        >>> save_model(model, encoders, scaler, 'models/my_model.pkl')
    """
    with open(filename, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "encoders": encoders,
                "scaler": scaler
            },
            f
        )


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


if __name__ == "__main__":

    df = pd.read_csv("datasets\penguins_size.csv")

    print("Training Decision Tree...")
    train_decision_tree(df)

    print("Training SVM...")
    train_svm(df)

    print("Training KNN...")
    train_knn(df)

    print("All models trained and saved successfully.")
