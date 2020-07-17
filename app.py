from flask import Flask, request, url_for, redirect, render_template, jsonify
import xgboost
import pandas as pd
import joblib


import numpy as np



app = Flask(__name__)
reg1 =joblib.load('Xgb.pkl')
cols = ['LSTAT','RM','PTRATIO','INDUS']


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = np.array(int_features)
    final1 = pd.DataFrame([final], columns=cols)
    prediction = reg1.predict(final1)

    return render_template('home.html', pred='Predicted House price will be {}'.format(prediction))


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = reg1.predict(data_unseen)

    return jsonify(prediction)


if __name__ == '__main__':
    app.run(debug=True)
