# main.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Load data
data = pd.read_csv("data/electricity.csv")

# ✅ Convert to datetime format
data["datetime"] = pd.to_datetime(data["datetime"], errors='coerce')

# ✅ Drop rows with invalid datetime
data.dropna(subset=["datetime"], inplace=True)

# ✅ Feature extraction
data["hour"] = data["datetime"].dt.hour
data["day"] = data["datetime"].dt.day
data["month"] = data["datetime"].dt.month
data["weekday"] = data["datetime"].dt.weekday

# Define features and label
X = data[["hour", "day", "month", "weekday"]]
y = data["consumption"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"📉 Mean Squared Error: {mse:.2f}")

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/forecast_model.pkl")
print("✅ Model saved at model/forecast_model.pkl")
