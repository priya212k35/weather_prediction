import pandas as pd
from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Function to safely load a model
def safe_load_model(file_path):
    if os.path.exists(file_path):
        try:
            return joblib.load(file_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print(f"Model file {file_path} not found")
        return None

# Load the models and encoders
model = safe_load_model('weather_model.pkl')
le_location = safe_load_model('model/le_location.pkl')
le_weather_type = safe_load_model('model/le_weather_type.pkl')
le_cloud_cover = safe_load_model('model/le_cloud_cover.pkl')
le_season = safe_load_model('model/le_season.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not all([le_location, le_weather_type, le_cloud_cover, le_season]):
        return "Model or encoders not loaded properly"

    # Get input data from the form
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    wind_speed = float(request.form['wind_speed'])
    precipitation = float(request.form['precipitation'])
    cloud_cover = request.form['cloud_cover']
    atmospheric_pressure = float(request.form['atmospheric_pressure'])
    uv_index = float(request.form['uv_index'])
    season = request.form['season']
    visibility = float(request.form['visibility'])
    location = request.form['location']

    # Encode categorical variables
    cloud_cover_encoded = le_cloud_cover.transform([cloud_cover])[0]
    season_encoded = le_season.transform([season])[0]
    location_encoded = le_location.transform([location])[0]

    # Create DataFrame for the input
    input_data = pd.DataFrame([[temperature, humidity, wind_speed, precipitation,
                                 cloud_cover_encoded, atmospheric_pressure,
                                 uv_index, season_encoded, visibility, location_encoded]],
                               columns=['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)',
                                        'Cloud Cover', 'Atmospheric Pressure', 'UV Index',
                                        'Season', 'Visibility (km)', 'Location'])

    # Make prediction
    try:
        prediction = model.predict(input_data)
        predicted_weather = le_weather_type.inverse_transform(prediction)[0]
    except Exception as e:
        return f"Error during prediction: {e}"

    return render_template('result.html', prediction=predicted_weather)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

