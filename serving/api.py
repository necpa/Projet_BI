from fastapi import FastAPI, HTTPException
import uvicorn
import pickle
import numpy as np
import pandas as pd
from pydantic import BaseModel

# Définition du modèle de requête attendu
class InputData(BaseModel):
    features: list

# Initialisation de l'API
app = FastAPI()

# Chargement des modèles et scaler
try:
    with open("artifacts/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("artifacts/embedding.pkl", "rb") as f:
        embedding_model = pickle.load(f)
    with open("artifacts/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement des modèles: {e}")

@app.post("/predict")
def predict(data: InputData):
    try:
        # Conversion des features en array numpy
        features = np.array(data.features).reshape(1, -1)
        
        # Transformation des données avec le modèle d’embedding et le scaler
        embedded_features = embedding_model.transform(features)
        scaled_features = scaler.transform(embedded_features)
        
        # Prédiction
        prediction = model.predict(scaled_features)
        
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction: {e}")

# Point d'entrée pour exécuter l'API en local
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
