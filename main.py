import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
data = {
    "Area": [1500, 1800, 2400, 3000, 3500, 4000, 4200, 4600, 5000, 5500],
    "Bedrooms": [3, 4, 3, 5, 4, 6, 5, 7, 6, 8],
    "Age": [5, 8, 10, 15, 7, 12, 9, 20, 18, 25],
    "Price": [300000, 360000, 480000, 600000, 700000, 720000, 750000, 840000, 900000, 980000]
}

df = pd.DataFrame(data)

# Features & Target Variable
X = df[["Area", "Bedrooms", "Age"]]
y = df["Price"]

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# ======================== Visualization ========================

#  Scatter Plot: Actual vs Predicted Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color="blue", edgecolors="black", alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="red")  # Perfect Prediction Line
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

#  Regression Plot: Price vs Area
plt.figure(figsize=(8, 6))
sns.regplot(x=df["Area"], y=df["Price"], scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.xlabel("House Area (sq ft)")
plt.ylabel("Price ($)")
plt.title("House Price vs Area")
plt.grid(True)
plt.show()

#  Correlation Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
