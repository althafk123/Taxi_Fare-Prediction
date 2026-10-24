
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("taxi_trip_pricing.csv", sep="\t")

# 2. Keep records with known target values
df = df.dropna(subset=["Trip_Price"]).copy()

# 3. Define input features and target
features = [
    "Trip_Distance_km",
    "Time_of_Day",
    "Day_of_Week",
    "Passenger_Count",
    "Traffic_Conditions",
    "Weather",
    "Trip_Duration_Minutes"
]

X = df[features]
y = df["Trip_Price"]

# 4. Separate numerical and categorical columns
numeric_features = [
    "Trip_Distance_km",
    "Passenger_Count",
    "Trip_Duration_Minutes"
]

categorical_features = [
    "Time_of_Day",
    "Day_of_Week",
    "Traffic_Conditions",
    "Weather"
]

# 5. Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 6. Build model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 8. Train model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 9. Predict
predictions = model.predict(X_test)

# 10. Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n===== MODEL EVALUATION =====")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.3f}")

# 11. Save trained model
joblib.dump(model, "taxi_fare_model.pkl")

print("\nModel saved as taxi_fare_model.pkl")
