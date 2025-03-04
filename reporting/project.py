import os
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

# Charger les données de référence et de production
ref_data = pd.read_csv("../data/ref_data.csv")
prod_data = pd.read_csv("../data/prod_data.csv")

# Vérifier et nettoyer les données
required_columns = ["target", "prediction", "age", "avg_glucose_level", "bmi", "gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status"]

for col in required_columns:
    if col not in ref_data.columns:
        print(f"Column {col} is missing in reference data, adding a placeholder column.")
        ref_data[col] = 0  # Ajouter une colonne fictive avec des valeurs par défaut
    if col not in prod_data.columns:
        print(f"Column {col} is missing in production data, adding a placeholder column.")
        prod_data[col] = 0  # Ajouter une colonne fictive avec des valeurs par défaut

# Définir le mapping des colonnes
column_mapping = ColumnMapping(
    target="target",
    prediction="prediction",
    numerical_features=["age", "avg_glucose_level", "bmi"],
    categorical_features=["gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status"]
)

# Créer le rapport Evidently
report = Report(metrics=[
    DataDriftPreset(),
    ClassificationPreset()
])
report.run(reference_data=ref_data, current_data=prod_data, column_mapping=column_mapping)

# Sauvegarder le rapport
report.save_html("/app/reporting/report.html")
