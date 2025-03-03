import os
import pandas as pd
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

API_URL = os.environ.get("API_URL","http://localhost:8080")
API_URL_SINGLE = API_URL + "/predict"
API_URL_BULK = API_URL + "/predict/all"
API_URL_FEEDBACK = API_URL + "/feedback"


@app.route("/", methods=["GET", "POST"])
def form():
    data = []
    error = None
    api_response = None  # Stockera la réponse de l'API

    if request.method == "POST":
        form_type = request.form.get("form_type")
        stroke_values = []
        # Si un fichier CSV est uploadé
        if form_type == "file_upload" and "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)  # Sauvegarde temporaire du fichier

            try:
                # Charger les données du CSV avant l'envoi
                df = pd.read_csv(file_path)
                data = df.to_dict(orient="records")  # Convertir en liste de dictionnaires

                # Vérifier si la colonne "stroke" existe
                if "stroke" in df.columns:
                    stroke_values = df["stroke"].tolist()  # Extraire les valeurs de la colonne "stroke"
                    df = df.drop(columns=["stroke"])  # Supprimer la colonne "stroke"

                # Sauvegarder un fichier temporaire sans la colonne "stroke"
                temp_file_path = os.path.join(UPLOAD_FOLDER, "temp_without_stroke.csv")
                df.to_csv(temp_file_path, index=False)  # Réécrire le CSV sans la colonne stroke

                # Envoyer le fichier modifié à l'API
                with open(temp_file_path, "rb") as f:
                    response = requests.post(API_URL_BULK, files={"file": f})

                api_response = response.json()
            except Exception as e:
                api_response = {"error": f"Erreur lors de l'envoi du fichier à l'API : {str(e)}"}

        # Si l'utilisateur remplit le formulaire à la main
        elif form_type == "manual_entry":
            form_data = request.form.to_dict()
            form_data.pop("form_type")

            # Ajouter les données pour affichage
            data = [form_data]

            try:
                # Envoyer les données à l'API
                response = requests.post(API_URL_SINGLE, json=form_data)
                api_response = response.json()
            except Exception as e:
                api_response = {"error": f"Erreur lors de l'envoi des données à l'API : {str(e)}"}
        print(data)
        return render_template("feedback.html", data=data,target = stroke_values ,api_response=api_response.get("prediction",[]), error= api_response.get("error", ""))

    return render_template("form.html")

@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        feedback_data = []

        # Récupération des clés du formulaire en excluant les colonnes target et prediction
        data_keys = ["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status", "target", "prediction"]
        # Nombre total de lignes dans le formulaire
        num_rows = len(request.form) // (len(data_keys) + 2)  # En prenant en compte target et prediction

        # Parcours de chaque ligne (chaque entrée)
        for i in range(num_rows):
            row_data = {}

            # Ajouter les valeurs des autres colonnes (toutes sauf target et prediction)
            for key in data_keys:
                row_data[key] = request.form.get(f"{key}_{i}", "")

            # Ajouter cette ligne complète dans la liste
            feedback_data.append(row_data)

        # Créer le payload conforme à la structure attendue par l'API
        payload = {"data": feedback_data}

        print(payload)

        # Envoyer le feedback à l'API
        response = requests.post(API_URL_FEEDBACK, json=payload)

        if response.status_code == 200:
            return jsonify({"success": "Feedback envoyé avec succès"}), 200
        else:
            return jsonify({"error": f"Erreur API: {response.text}"}), 400

    except Exception as e:
        return jsonify({"error": f"Erreur lors de l'envoi du feedback : {str(e)}"}), 500




if __name__ == "__main__":
    app.run(debug=True)
