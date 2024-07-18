from flask import Flask, render_template, request
import pickle
import joblib
import pandas as pd
import os
import json

app = Flask(__name__)
model = pickle.load(open("adaboost_model.pkl", "rb"))

file_path = os.path.join(os.path.dirname(__file__), "feature_values.json")
ct = file_path

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/index.html')
def predict():
    return render_template("index.html")

@app.route('/output.html', methods=["GET", "POST"])
def output():
    result_message = None

    if request.method == "POST":
        age = request.form["age"]
        gender = request.form["gender"]
        family_history = request.form["family_history"]
        benefits = request.form["benefits"]
        care_options = request.form["care_options"]
        anonymity = request.form["anonymity"]
        leave = request.form["Leave"]
        work_interfere = request.form["work_interfere"]

        data = [[age, gender, family_history, work_interfere, benefits, care_options, anonymity, leave]]

        feature_cols = ['Age', 'Gender', 'family history', 'work_interfere', 'benefits', 'care_options', 'anonymity', 'Leave']

        with open(ct, 'r') as file:
            json_data = json.load(file)

        user_data = pd.DataFrame([json_data])

        pred = model.predict(user_data)
        pred = pred[0]

        result_message = "This person requires mental health treatment" if pred else "This person doesn't require mental health treatment"

    return render_template("output.html", result_message=result_message)

if __name__ == '__main__':
    app.run(debug=True)
