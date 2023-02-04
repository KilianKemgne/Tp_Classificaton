import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load



# Create flask app
flask_app = Flask(__name__)

vector = load("vectors.joblib")
model = load("model.joblib")

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/evaluation", methods = ["POST"])
def predict():
    print(request.form.get("text"))
    text=[request.form.get("text")]
    vec = vector.transform(text)
    prediction = model.predict(vec)
    prediction = int(prediction)
    if prediction >0:
        prediction="Positif 🙃"
    else:
        prediction = "Negatif 😞"
    return render_template("index.html", resultat = "Le statut du commentaire est : {} ".format(prediction))


if __name__ == "__main__":
    flask_app.run(debug=True)