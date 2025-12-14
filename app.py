from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("crop_model.pkl")

@app.route("/")
def home():
    return "Crop Recommendation API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        x = np.array([
            data["N"],
            data["P"],
            data["K"],
            data["temperature"],
            data["humidity"],
            data["ph"],
            data["rainfall"]
        ]).reshape(1, -1)

        crop = model.predict(x)[0]   # ✅ NO label encoder

        return jsonify({"crop": crop})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

