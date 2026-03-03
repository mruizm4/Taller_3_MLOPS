import pandas as pd
from train import load_model
from predict import predict_new_data

# Cargar modelo (prueba con svm, knn o decision_tree)
model, encoders, scaler = load_model("models/decision_tree.pkl")


# Crear un DF con 1 registro
df_test = pd.DataFrame([{
    "culmen_length_mm": 39.1,
    "culmen_depth_mm": 18.7,
    "flipper_length_mm": 181,
    "body_mass_g": 3750,
    "island": "Torgersen",
    "sex": "Male"
}])

# Ahora predict_new_data usa OneHotEncoder para variables categóricas
prediction = predict_new_data(df_test, model, encoders, scaler)

print("Prediction:", prediction)
