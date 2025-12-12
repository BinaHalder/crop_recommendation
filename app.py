from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

# Initialize app
app = Flask(__name__)
CORS(app)

# Load model and label encoder
model = joblib.load("crop_model.pkl")


# Home route
@app.route("/")
def home():
    return "Crop Recommendation API is running"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    x = np.array([data[f] for f in features]).reshape(1, -1)

    pred = model.predict(x)[0]
 

    return jsonify({"crop": label})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)


