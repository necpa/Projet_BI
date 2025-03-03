import os
import pandas as pd
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

API_URL_SINGLE = "http://serving-api:8080/predict"
API_URL_BULK = "http://serving-api:8080/predict/all"


@app.route("/", methods=["GET", "POST"])
def form():
    data = []
    api_response = None  # Stockera la réponse de l'API

    if request.method == "POST":
        form_type = request.form.get("form_type")

        # Si un fichier CSV est uploadé
        if form_type == "file_upload" and "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)  # Sauvegarde temporaire du fichier

            try:
                # Charger les données du CSV avant l'envoi
                df = pd.read_csv(file_path)
                data = df.to_dict(orient="records")  # Convertir en liste de dictionnaires

                # Envoyer le fichier à l'API
                with open(file_path, "rb") as f:
                    response = requests.post(API_URL_BULK, files={"file": f})

                api_response = response.json()
            except Exception as e:
                api_response = {"error": f"Erreur lors de l'envoi du fichier à l'API : {str(e)}"}

        # Si l'utilisateur remplit le formulaire à la main
        elif form_type == "manual_entry":
            form_data = request.form.to_dict()

            # Nettoyage des types (conversion en int / float pour correspondre au modèle FastAPI)
            form_data["age"] = int(form_data["age"])
            form_data["hypertension"] = int(form_data["hypertension"])
            form_data["heart_disease"] = int(form_data["heart_disease"])
            form_data["avg_glucose_level"] = float(form_data["avg_glucose_level"])
            form_data["bmi"] = float(form_data["bmi"]) if form_data["bmi"] else None

            # Ajouter les données pour affichage
            data = [form_data]

            try:
                # Envoyer les données à l'API
                response = requests.post(API_URL_SINGLE, json=form_data)
                api_response = response.json()
            except Exception as e:
                api_response = {"error": f"Erreur lors de l'envoi des données à l'API : {str(e)}"}

        return render_template("result.html", data=data, api_response=api_response)

    return render_template("form.html")


if __name__ == "__main__":
    app.run(debug=True)
