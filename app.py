from flask import Flask, request, jsonify
import joblib
import numpy as np

from flask_cors import CORS
CORS(app)

app = Flask(__name__)

model = joblib.load("crop_model.pkl")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = ['N','P','K','temperature','humidity','ph','rainfall']
    x = np.array([data[f] for f in features]).reshape(1,-1)
    pred = model.predict(x)[0]
    label = le.inverse_transform([pred])[0]
    return jsonify({"recommended_crop": label})

if __name__ == '__main__':
    app.run()

