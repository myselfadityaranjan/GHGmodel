from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os
import json

# Import your functions from src
from src.data_fetch import fetch_ghg_data
from src.preprocessing import preprocess_data
from src.model_train import train_model
from src.optimization import recommend_land_use_changes

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load city data
with open('data/cities.json') as f:
    cities = json.load(f)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html', cities=cities)

# Route to fetch GHG data and preprocess it
@app.route('/fetch_data', methods=['POST'])
def fetch_data():
    try:
        content = request.get_json()
        lat = content['latitude']
        lon = content['longitude']
        fetch_ghg_data()  # Fetch GHG data from the source
        df = preprocess_data()  # Preprocess the fetched data
        return jsonify({"message": f"Data fetched for location ({lat}, {lon})!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to train the model
@app.route('/train_model', methods=['POST'])
def train():
    try:
        data = np.load('data/preprocessed_data.npy', allow_pickle=True).item()
        X_train = data['X_train']
        y_train = data['y_train']
        model = train_model(X_train, y_train)  # Train the model
        joblib.dump(model, 'data/trained_model.pkl')  # Save the trained model
        return jsonify({"message": "Model trained successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to get land use recommendations based on latitude and longitude
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        lat = request.json['latitude']
        lon = request.json['longitude']
        df = preprocess_data()  # Ensure data is preprocessed
        
        recommendations = recommend_land_use_changes(df, lat, lon)
        filtered_recommendations = recommendations[:5]  # Example: top 5 recommendations
        
        return jsonify({"recommendations": filtered_recommendations}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
