# api.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model (7 inputs: px, py, pz, rx, ry, rz, depth)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Support both GET and POST
    if request.method == "GET":
        data = request.args
    else:
        data = request.json

    try:
        px = float(data.get("px", 0))
        py = float(data.get("py", 0))
        pz = float(data.get("pz", 0))
        rx = float(data.get("rx", 0))
        ry = float(data.get("ry", 0))
        rz = float(data.get("rz", 0))
        depth = float(data.get("depth", 0))

        # Build input array
        input_data = np.array([[px, py, pz, rx, ry, rz, depth]])
        prob = model.predict_proba(input_data)[0][1]

        result = "pass" if prob > 0.5 else "fail"

        return jsonify({
            "result": result,
            "confidence": float(prob),
            "hasConfidence": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Use 0.0.0.0 to allow LAN access (Unity can hit it via IP)
    app.run(host="0.0.0.0", port=5000)