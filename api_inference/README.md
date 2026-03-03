

# 🐧 Taller_1_MLOPS_PUJ: Penguin Prediction API. 

---
Por: Carlos Carvajales y Mateo Ruiz

API REST desarrollada con **FastAPI** para realizar predicciones de especie de pingüino utilizando múltiples modelos de Machine Learning previamente entrenados.

La API permite seleccionar dinámicamente uno o varios modelos para obtener predicciones en una única petición.

---

## 🚀 Modelos Disponibles

* `TREE` → Decision Tree
* `KNN` → K-Nearest Neighbors
* `SVM` → Support Vector Machine

Cada modelo fue entrenado previamente y serializado (`.pkl`).
Se cargan automáticamente al iniciar la aplicación.

---

## 📡 Endpoint Principal

### `POST /predict`

Genera predicciones en función de las características físicas enviadas como parámetros `Query`.

### Parámetros

| Parámetro           | Tipo       | Descripción                                        |
| ------------------- | ---------- | -------------------------------------------------- |
| `models`            | List[Enum] | Lista de modelos a utilizar (`TREE`, `KNN`, `SVM`) |
| `culmen_length_mm`  | float      | Longitud del pico                                |
| `culmen_depth_mm`   | float      | Profundidad del pico                             |
| `flipper_length_mm` | float      | Longitud de la aleta                               |
| `body_mass_g`       | float      | Masa corporal en gramos                            |
| `island`            | Enum       | Isla de origen                                     |
| `sex`               | Enum       | Sexo del ejemplar                                  |

---

### Ejemplo de Request

```
POST /predict?models=TREE&models=SVM&culmen_length_mm=39&culmen_depth_mm=18.7&flipper_length_mm=180&body_mass_g=3700&island=Torgersen&sex=Male
```

---

### Ejemplo de Response

```json
{
  "TREE": ["Adelie"],
  "SVM": ["Adelie"]
}
```

---

## 🐳 Ejecución con Docker

### Construir la imagen

```bash
docker build -t taller_1_image .
```

### Ejecutar el contenedor

```bash
docker run --name taller_1 -p 8000:8000 taller_1_image
```

La API quedará disponible en:

```
http://localhost:8000/docs
```

---

## 📂 Estructura del Proyecto

```
├── datasets/
├── models/
├── models_performance/
├── main.py
├── predict.py
├── train.py
├── test_predict.py
├── Dockerfile
├── requirements.txt
└── README.md
```

> `train.py` y `test_predict.py` no forman parte de la API en ejecución.
> Se incluyen únicamente como referencia para entrenamiento y pruebas locales de predicción.

---

## ⚙️ Consideraciones Técnicas

* Los modelos se cargan una sola vez al iniciar la aplicación.
* Validación automática mediante Pydantic.
* Soporte para múltiples modelos en una misma solicitud.
* Contenedor Docker listo para despliegue.

