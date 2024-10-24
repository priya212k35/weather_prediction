import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load the dataset
file_path = 'data/weather_data.csv'  # Path to your dataset
df = pd.read_csv(file_path)

# Label encode categorical variables
le_location = LabelEncoder()
le_weather_type = LabelEncoder()
le_cloud_cover = LabelEncoder()
le_season = LabelEncoder()

df['Location'] = le_location.fit_transform(df['Location'])
df['Weather Type'] = le_weather_type.fit_transform(df['Weather Type'])
df['Cloud Cover'] = le_cloud_cover.fit_transform(df['Cloud Cover'])
df['Season'] = le_season.fit_transform(df['Season'])

# Split data into features (X) and target (y)
X = df.drop('Weather Type', axis=1)
y = df['Weather Type']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Save the model and encoders
joblib.dump(model, 'weather_model.pkl')
joblib.dump(le_location, 'model/le_location.joblib')
joblib.dump(le_weather_type, 'model/le_weather_type.joblib')
joblib.dump(le_cloud_cover, 'model/le_cloud_cover.joblib')
joblib.dump(le_season, 'model/le_season.joblib')

print("Model and encoders saved.")
