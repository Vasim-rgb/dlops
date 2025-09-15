from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load model and encoders/scaler (ensure these files are in same folder)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
LE_GENDER_PATH = os.path.join(os.path.dirname(__file__), "label_encoder_gender.pkl")
OHE_GEO_PATH = os.path.join(os.path.dirname(__file__), "onehot_encoder_geo.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

model = tf.keras.models.load_model(MODEL_PATH)
with open(LE_GENDER_PATH, "rb") as f:
    label_encoder_gender = pickle.load(f)
with open(OHE_GEO_PATH, "rb") as f:
    onehot_encoder_geo = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    # provide choices for select boxes using loaded encoders
    gender_choices = list(label_encoder_gender.classes_)
    geo_choices = onehot_encoder_geo.categories_[0].tolist()
    return render_template("index.html", genders=gender_choices, geos=geo_choices)

@app.route("/predict", methods=["POST"])
def predict():
    # read form values
    geography = request.form.get("geography")
    gender = request.form.get("gender")
    age = int(request.form.get("age", 30))
    balance = float(request.form.get("balance", 0.0) or 0.0)
    credit_score = float(request.form.get("credit_score", 500) or 0.0)
    estimated_salary = float(request.form.get("estimated_salary", 0.0) or 0.0)
    tenure = int(request.form.get("tenure", 0))
    num_of_products = int(request.form.get("num_of_products", 1))
    has_cr_card = int(request.form.get("has_cr_card", 0))
    is_active_member = int(request.form.get("is_active_member", 0))

    # prepare dataframe similar to your streamlit logic
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # one-hot encode geography
    geo_arr = onehot_encoder_geo.transform([[geography]])
    # If sparse matrix returned, convert to array
    try:
        geo_encoded = geo_arr.toarray()
    except Exception:
        geo_encoded = np.asarray(geo_arr)
    geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    input_combined = pd.concat([input_data.reset_index(drop=True), geo_df.reset_index(drop=True)], axis=1)

    # scale
    input_scaled = scaler.transform(input_combined)

    # predict
    pred = model.predict(input_scaled)
    # handle shape (1,1) or (1,) etc
    try:
        proba = float(pred[0][0])
    except Exception:
        proba = float(np.ravel(pred)[0])

    churn_flag = proba > 0.5

    # pass results to a result template
    return render_template("result.html",
                           probability=round(proba, 4),
                           churn=churn_flag,
                           age=age,
                           credit_score=credit_score,
                           balance=balance,
                           estimated_salary=estimated_salary)

if __name__ == "__main__":
    app.run(debug=True)