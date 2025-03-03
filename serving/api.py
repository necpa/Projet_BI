from fastapi import FastAPI, HTTPException, UploadFile
import pandas as pd
import uvicorn
import joblib
import numpy as np
from pydantic import BaseModel

# Définition du modèle de requête attendu
class FormData(BaseModel):
    gender: str
    age: int
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float
    bmi: float | None = None
    smoking_status: str


# Initialisation de l'API
app = FastAPI()

# Chargement des modèles et scaler
try:
    with open("../artifacts/model.joblib", "rb") as f:
        model = joblib.load(f)
    with open("../artifacts/embedding.joblib", "rb") as f:
        embedding_model = joblib.load(f)
    with open("../artifacts/scaler.joblib", "rb") as f:
        scaler = joblib.load(f)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement des modèles: {e}")

@app.post("/predict")
async def predict(data: FormData):
    try:
        # Conversion de InpuData en numpy Array
        dataArray = np.array([list(data.model_dump().values())])
        print(dataArray)
        
        # Transformation des données avec le modèle d’embedding et le scaler
        transfrom_data = embedding_model.transform(dataArray)
        print(transfrom_data)

        scaled_data = scaler.transform(transfrom_data)
        print(scaled_data)
        
        # Prédiction
        prediction = model.predict(scaled_data)
        print(prediction)
        
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {e}")


@app.post("/predict/all")
async def predict(file: UploadFile):
    try:
        # Conversion de InpuData en numpy Array
        data = pd.read_csv(file.file)
        dataArray = data.values
        print(dataArray)

        # Transformation des données avec le modèle d’embedding et le scaler
        transfrom_data = embedding_model.transform(dataArray)
        print(transfrom_data)

        scaled_data = scaler.transform(transfrom_data)
        print(scaled_data)

        # Prédiction
        prediction = model.predict(scaled_data)
        print(prediction)

        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {e}")

# Point d'entrée pour exécuter l'API en local
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
