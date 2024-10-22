import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load your dataset (update the path as necessary)
df = pd.read_csv('data/weather_data.csv')  # Change this path to your dataset

# Initialize Label Encoders
le_location = LabelEncoder()
le_weather_type = LabelEncoder()
le_cloud_cover = LabelEncoder()
le_season = LabelEncoder()

# Fit the label encoders on the respective columns
le_location.fit(df['Location'])
le_weather_type.fit(df['Weather Type'])
le_cloud_cover.fit(df['Cloud Cover'])
le_season.fit(df['Season'])

# Create the model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Save the label encoders
joblib.dump(le_location, 'model/le_location.joblib')
joblib.dump(le_weather_type, 'model/le_weather_type.joblib')
joblib.dump(le_cloud_cover, 'model/le_cloud_cover.joblib')
joblib.dump(le_season, 'model/le_season.joblib')

