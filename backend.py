# backend.py

from flask import Flask, request, jsonify
import pickle
from datetime import datetime

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

@app.route("/")
def home():
    return "Backend Running ✅"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    material = data["material"]
    weight = data["weight"]
    demand = data["demand"]

    material_encoded = encoder.transform([material])[0]

    hour = datetime.now().hour
    day = datetime.now().weekday()

    prediction = model.predict([[material_encoded, weight, demand, hour, day]])

    return jsonify({"price": float(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    