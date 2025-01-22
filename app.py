from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # Tambahkan ini untuk mengaktifkan CORS

# Load model dan pipeline
model = joblib.load('random_forest_model.pkl')
pipeline = joblib.load('preprocessing_pipeline.pkl')

@app.route('/')
def home():
    return "API untuk Prediksi Penyakit Jantung. Gunakan endpoint /predict untuk memprediksi."

@app.route('/favicon.ico')
def favicon():
    return "", 204

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari request
        data = request.get_json()

        # Nama kolom sesuai dataset
        columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                   'restecg', 'thalach', 'exang', 'oldpeak', 
                   'slope', 'ca', 'thal']

        # Convert ke DataFrame
        input_data = pd.DataFrame([data], columns=columns)

        # Proses data dengan pipeline
        processed_data = pipeline.transform(input_data)

        # Prediksi menggunakan model
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)

        # Buat respons
        result = {
            "prediction": int(prediction[0]),
            "probability": {
                "no_disease": float(probability[0][0]),
                "disease": float(probability[0][1])
            }
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
