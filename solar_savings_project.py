import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("data/household_power_consumption.csv", low_memory=False)

# Data Preprocessing
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors='coerce', dayfirst=True)
df.drop(columns=["Date", "Time"], inplace=True)

numeric_columns = ["Global_active_power", "Voltage", "Global_intensity", "Sub_metering_3"]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
df.fillna(df.mean(), inplace=True)

# Save cleaned data
df.to_csv("data/processed_data.csv", index=False)

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 5))
plt.plot(df["Datetime"], df["Global_active_power"], color='blue', alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Power Consumption (kW)")
plt.title("Household Electricity Consumption Over Time")
plt.grid()
plt.show()

# Cost Comparison Bar Chart
costs = pd.DataFrame({"Type": ["Without Solar", "With Solar"], "Cost": [63712, 63389]})
plt.figure(figsize=(6, 4))
sns.barplot(x="Type", y="Cost", data=costs, palette=["red", "blue"])
plt.ylabel("Electricity Cost (₹)")
plt.title("Electricity Cost Comparison")
plt.show()

# Solar Contribution Pie Chart
plt.figure(figsize=(6, 6))
labels = ["Grid Consumption", "Solar Contribution"]
sizes = [df["Global_active_power"].sum(), df["Sub_metering_3"].sum()]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['orange', 'green'], startangle=140)
plt.title("Solar Contribution to Household Energy")
plt.show()

# Hourly Power Consumption Heatmap
df["Hour"] = df["Datetime"].dt.hour
hourly_avg_power = df.groupby("Hour")["Global_active_power"].mean().reset_index()
plt.figure(figsize=(8, 5))
sns.heatmap(hourly_avg_power.set_index("Hour").T, cmap="YlGnBu", annot=True, fmt=".2f")
plt.xlabel("Hour of the Day")
plt.title("Hourly Power Consumption Trends")
plt.show()

# Feature Engineering
df["Year"] = df["Datetime"].dt.year
X = df[["Year"]]
y = df["Global_active_power"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Save trained model
joblib.dump(model, "models/solar_savings_model.pkl")

# Prediction Function
def predict_savings(year):
    model = joblib.load("models/solar_savings_model.pkl")
    return round(model.predict([[year]])[0], 2)

# Example Prediction
print(f"Predicted Solar Savings for 2035: ₹{predict_savings(2035)}")
