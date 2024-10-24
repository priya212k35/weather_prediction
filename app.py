import pandas as pd
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('weather_model.pkl')

# Load the label encoders
le_location = joblib.load('model/le_location.joblib')
le_weather_type = joblib.load('model/le_weather_type.joblib')
le_cloud_cover = joblib.load('model/le_cloud_cover.joblib')
le_season = joblib.load('model/le_season.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
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
    prediction = model.predict(input_data)
    predicted_weather = le_weather_type.inverse_transform(prediction)[0]

    return render_template('result.html', prediction=predicted_weather)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

