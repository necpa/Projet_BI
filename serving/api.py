from typing import List

from fastapi import FastAPI, HTTPException, UploadFile
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
    Residence_type: str
    avg_glucose_level: float
    bmi: float | None = None
    smoking_status: str

class FeedbackData(BaseModel):
    gender: str
    age: int
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float | None = None
    smoking_status: str
    target: int
    prediction: int

class FeedbackDatas(BaseModel):
    data: List[FeedbackData]

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

RETRAIN_THRESHOLD = 100

def retrain_model():
    global model
    try:
        ref_data = pd.read_csv("../data/ref_data.csv")
        prod_data = pd.read_csv("../data/prod_data.csv")
        combined_data = pd.concat([ref_data, prod_data], ignore_index=True)

        X = combined_data.drop(columns=["target", "prediction"])
        y = combined_data["target"]

        model = RandomForestClassifier()
        model.fit(X, y)

        joblib.dump(model, "../artifacts/model.joblib")

        model = model
        print("Model retrained and updated successfully.")
    except Exception as e:
        raise RuntimeError(f"Erreur lors du réentraînement du modèle: {e}")
    

@app.post("/predict")
async def predict(data: FormData):
    try:
        # Conversion de InpuData en numpy Array
        dataArray = np.array([list(data.model_dump().values())])
        
        # Transformation des données avec le modèle d’embedding et le scaler
        transfrom_data = embedding_model.transform(dataArray)

        scaled_data = scaler.transform(transfrom_data)
        
        # Prédiction
        prediction = model.predict(scaled_data)
        
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {e}")


@app.post("/predict/all")
async def predict(file: UploadFile):
    try:
        # Conversion de InpuData en numpy Array
        data = pd.read_csv(file.file)
        dataArray = data.values

        # Transformation des données avec le modèle d’embedding et le scaler
        transfrom_data = embedding_model.transform(dataArray)

        scaled_data = scaler.transform(transfrom_data)

        # Prédiction
        prediction = model.predict(scaled_data)

        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {e}")

@app.post("/feedback")
def feedback(data: FeedbackDatas):
    try:
        print(data)
        dataArray = np.array([list(k.model_dump().values()) for k in data.data])
        print(dataArray)
        x = dataArray[:,:-2]
        y = dataArray[:,-2:]
        x_embedded = embedding_model.transform(x)
        x_scaled = scaler.transform(x_embedded)

        with open("../data/prod_data.csv", "a") as file:
            for row in range(len(x_scaled)):
                for val in x_scaled[row]:
                    file.write(str(val))
                    file.write(",")
                file.write(str(y[row][0]) + "," + str(y[row][1]) + "\n")
                print(f"Row written: {x_scaled[row]}, {y[row][0]}, {y[row][1]}")  # Log each row written


        # Check if retraining is needed
        prod_data = pd.read_csv("../data/prod_data.csv")
        if len(prod_data) % RETRAIN_THRESHOLD == 0:
            retrain_model()

        return {"feedback": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du feedback: {e}")


@app.get("/feedback/data")
def get_feedback_data():
    try:
        with open("../data/prod_data.csv", "r") as file:
            content = file.read()
        return {"data": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la lecture du fichier: {e}")
    

# Point d'entrée pour exécuter l'API en local
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
