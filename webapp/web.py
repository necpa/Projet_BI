import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def form():
    data = []
    if request.method == "POST":
        form_type = request.form.get("form_type")


        # Vérifier si un fichier a été uploadé
        if form_type == "file_upload" and "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]

            try:
                # Lire le fichier CSV
                df = pd.read_csv(file)  # On peut ajouter sep=";" si le fichier utilise des ";"
                # Convertir en liste de dictionnaires pour affichage
                data = df.to_dict(orient="records")
            except Exception as e:
                return f"Erreur lors de la lecture du fichier : {str(e)}"


        # Vérifier si le formulaire manuel est soumis
        elif form_type == "manual_entry":
            form_data = request.form.to_dict()
            data.append(form_data)

        return render_template("result.html", data=data)

    return render_template("form.html")


if __name__ == "__main__":
    app.run(debug=True)
