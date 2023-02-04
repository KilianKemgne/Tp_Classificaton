from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder="templates")
train_df = pd.read_csv("imdb_train.csv")
vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
vectorizer.fit_transform(train_df.review)


def ValuePredictor(to_predict_list):
    to_predict = vectorizer.transform(to_predict_list)
    loaded_model = pickle.load(open("bayn_model.pkl", "rb"))
    # to_predict = vectorizer.transform(to_predict)
    result = loaded_model.predict(to_predict)

    return result[0]


@app.route("/")
def home():
    return render_template("index.html")


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
