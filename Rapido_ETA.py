import pandas as pd
import numpy as np

# Number of samples
num_samples = 5000

# List of top places in Hyderabad (for Source & Destination)
places = [
    "Charminar", "Hitech City", "Gachibowli", "Banjara Hills", "Secunderabad",
    "Tank Bund", "Kukatpally", "Ameerpet", "Jubilee Hills", "Dilsukhnagar"
]

# Generate random dataset
data = {
    "source": np.random.choice(places, num_samples),
    "destination": np.random.choice(places, num_samples),
    "distance": np.random.uniform(2, 30, num_samples),  # Distance in km
    "avg_speed": np.random.uniform(5, 80, num_samples),  # Speed in km/h
    "traffic_density": np.random.uniform(0, 10, num_samples),  # 0 (Low) - 10 (High)
    "temperature": np.random.uniform(10, 45, num_samples),  # Celsius
    "humidity": np.random.uniform(20, 100, num_samples),  # %
    "rain_intensity": np.random.uniform(0, 1, num_samples),  # 0 (No Rain) - 1 (Heavy Rain)
    "day_of_week": np.random.randint(0, 7, num_samples),  # 0 (Monday) - 6 (Sunday)
    "hour": np.random.randint(0, 24, num_samples),  # 0-23 (Time of Day)
    "num_traffic_signals": np.random.randint(0, 10, num_samples),  # No. of traffic lights
    "road_condition": np.random.randint(0, 3, num_samples),  # 0 (Good), 1 (Moderate), 2 (Bad)
    "ride_type": np.random.randint(0, 2, num_samples)  # 0 (Economy), 1 (Premium)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Calculate ETA (Time = Distance / Speed * Traffic Factor)
df["eta"] = (df["distance"] / df["avg_speed"]) * (1 + df["traffic_density"] / 10)

# Save dataset as CSV
df.to_csv("eta_hyderabad_dataset.csv", index=False)

print("Dataset created and saved as eta_hyderabad_dataset.csv!")
print(df.head())


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("eta_hyderabad_dataset.csv")

# Define features and target
features = ["distance", "avg_speed", "traffic_density", "temperature", "humidity", 
            "rain_intensity", "day_of_week", "hour", "num_traffic_signals", 
            "road_condition", "ride_type"]

# One-hot encoding for categorical variables (source, destination)
encoder = OneHotEncoder(handle_unknown="ignore")
encoded_features = encoder.fit_transform(df[["source", "destination"]]).toarray()
encoded_feature_names = encoder.get_feature_names_out(["source", "destination"])

# Combine encoded categorical and numerical features
X = np.hstack((df[features], encoded_features))
y = df["eta"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the ML model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:\nMAE: {mae:.2f} minutes\nR² Score: {r2:.2f}")

# Save the model
joblib.dump(model, "eta_prediction_model.pkl")
joblib.dump(encoder, "encoder.pkl")



import joblib
import numpy as np

# Load the trained model and encoder
model = joblib.load("eta_prediction_model.pkl")
encoder = joblib.load("encoder.pkl")

# Function to predict ETA
def predict_eta(source, destination, distance, avg_speed, traffic_density, temperature, humidity, 
                rain_intensity, day_of_week, hour, num_traffic_signals, road_condition, ride_type):
    
    # One-hot encode the source and destination
    encoded_input = encoder.transform([[source, destination]]).toarray()
    
    # Prepare the input features
    features = np.hstack(([distance, avg_speed, traffic_density, temperature, humidity, 
                           rain_intensity, day_of_week, hour, num_traffic_signals, road_condition, ride_type], 
                          encoded_input.flatten()))
    
    # Predict ETA
    eta = model.predict([features])[0]
    return round(eta, 2)

# Example usage
source = "Charminar"
destination = "Gachibowli"
distance = 18  # km
avg_speed = 35  # km/h
traffic_density = 5  # Medium traffic
temperature = 30  # Celsius
humidity = 60  # %
rain_intensity = 0.2  # Light rain
day_of_week = 2  # Wednesday
hour = 17  # 5 PM
num_traffic_signals = 5
road_condition = 1  # Moderate
ride_type = 0  # Economy

eta = predict_eta(source, destination, distance, avg_speed, traffic_density, temperature, 
                  humidity, rain_intensity, day_of_week, hour, num_traffic_signals, road_condition, ride_type)

print(f"Estimated Time of Arrival (ETA): {eta} minutes")




