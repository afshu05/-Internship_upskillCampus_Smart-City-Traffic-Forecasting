# ============================================
# SMART CITY TRAFFIC PATTERN FORECASTING
# ============================================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --------------------------------------------
# Step 1: Load Dataset
# --------------------------------------------
train = pd.read_csv("train_aWnotuB.csv")
test = pd.read_csv("test_BdBKkAj.csv")

print("Train Data Sample:")
print(train.head())

# --------------------------------------------
# Step 2: Data Preprocessing
# --------------------------------------------

# Convert DateTime
train['DateTime'] = pd.to_datetime(train['DateTime'])
test['DateTime'] = pd.to_datetime(test['DateTime'])

# Extract features
for df in [train, test]:
    df['Hour'] = df['DateTime'].dt.hour
    df['Day'] = df['DateTime'].dt.day
    df['Month'] = df['DateTime'].dt.month
    df['Year'] = df['DateTime'].dt.year
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek

# --------------------------------------------
# Step 3: Add Weekend Feature (important)
# --------------------------------------------
train['Weekend'] = train['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
test['Weekend'] = test['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# --------------------------------------------
# Step 4: Features & Target
# --------------------------------------------
features = ['Hour', 'Day', 'Month', 'Year', 'DayOfWeek', 'Weekend', 'Junction']

X = train[features]
y = train['Vehicles']

# --------------------------------------------
# Step 5: Train-Test Split
# --------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------
# Step 6: Model Training
# --------------------------------------------
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------
# Step 7: Model Evaluation
# --------------------------------------------
y_pred = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("RMSE:", rmse)

accuracy = model.score(X_val, y_val)
print("Model Accuracy:", accuracy)

# --------------------------------------------
# Step 8: Visualization
# --------------------------------------------
plt.figure()
plt.scatter(y_val, y_pred)
plt.xlabel("Actual Traffic")
plt.ylabel("Predicted Traffic")
plt.title("Actual vs Predicted Traffic")
plt.show()

# --------------------------------------------
# Step 9: Traffic Pattern Visualization
# --------------------------------------------
plt.figure()
train.groupby('Hour')['Vehicles'].mean().plot()
plt.title("Average Traffic by Hour")
plt.xlabel("Hour")
plt.ylabel("Vehicles")
plt.show()

# --------------------------------------------
# Step 10: Prediction on Test Data
# --------------------------------------------
X_test_final = test[features]
test['Predicted_Traffic'] = model.predict(X_test_final)

print("\nPredicted Output Sample:")
print(test[['DateTime', 'Junction', 'Predicted_Traffic']].head())

# --------------------------------------------
# Step 11: Save Submission File
# --------------------------------------------
output = test[['ID', 'Predicted_Traffic']]
output.to_csv("submission.csv", index=False)

print("\n✅ submission.csv created successfully!")

# --------------------------------------------
# Step 12: Future Prediction Example
# --------------------------------------------
future_input = pd.DataFrame({
    'Hour': [18],
    'Day': [15],
    'Month': [6],
    'Year': [2024],
    'DayOfWeek': [2],
    'Weekend': [0],
    'Junction': [2]
})

future_prediction = model.predict(future_input)
print("\nFuture Traffic Prediction:", future_prediction[0])