from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("crop_model.pkl")

def validate_input(N, P, K, temperature, humidity, ph, rainfall):
    if not (0 <= N <= 140): return False
    if not (0 <= P <= 145): return False
    if not (0 <= K <= 205): return False
    if not (0 <= temperature <= 50): return False
    if not (0 <= humidity <= 100): return False
    if not (0 <= ph <= 14): return False
    if not (0 <= rainfall <= 300): return False
    return True


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        N = data["N"]
        P = data["P"]
        K = data["K"]
        temperature = data["temperature"]
        humidity = data["humidity"]
        ph = data["ph"]
        rainfall = data["rainfall"]

        if not validate_input(N, P, K, temperature, humidity, ph, rainfall):
            return jsonify({
                "error": "Invalid input values. Please enter realistic agricultural data."
            }), 400

        x = np.array([
            N, P, K, temperature, humidity, ph, rainfall
        ]).reshape(1, -1)

        crop = model.predict(x)[0]
        return jsonify({"crop": crop})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


