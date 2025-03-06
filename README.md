🏡 House Price Prediction using Linear Regression
A simple machine learning project that predicts house prices based on area, number of bedrooms, and house age using Linear Regression with NumPy, Pandas, and Scikit-Learn.

📌 Features
✅ Predicts house prices using Linear Regression
✅ Uses NumPy & Pandas for data processing
✅ Visualizes results with Matplotlib & Seaborn
✅ Evaluates performance using MAE, MSE, and R² Score

🛠 Technologies Used
Python 3.x
NumPy (for numerical computations)
Pandas (for data manipulation)
Matplotlib & Seaborn (for data visualization)
Scikit-Learn (for machine learning)
📥 Installation & Setup
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
2️⃣ Install Dependencies
bash
Copy
Edit
pip install numpy pandas matplotlib seaborn scikit-learn
3️⃣ Run the Project
bash
Copy
Edit
python house_price_prediction.py
📊 How It Works?
Loads a sample dataset containing Area, Bedrooms, Age, and Price.
Splits data into training (80%) and testing (20%) sets.
Trains a Linear Regression model to find the best relationship between features and price.
Makes predictions and evaluates using MAE, MSE, and R² Score.
Plots actual vs predicted prices for visualization.
📸 Visualizations
🔹 1. Scatter Plot: Actual vs Predicted Prices
Shows how well the model predicts house prices.

python
Copy
Edit
plt.scatter(y_test, y_pred, color="blue")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()
🔹 2. Regression Plot: Price vs Area
Displays price trends based on area size.

python
Copy
Edit
sns.regplot(x=df["Area"], y=df["Price"])
🔹 3. Correlation Heatmap
Shows relationships between features.

python
Copy
Edit
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
📄 Model Evaluation Metrics
Metric	Explanation
MAE (Mean Absolute Error)	Average absolute difference between actual and predicted prices.
MSE (Mean Squared Error)	Penalizes larger errors more.
R² Score	Measures how well the model explains the data (closer to 1 is better).
📄 License
This project is licensed under the MIT License – you can use, modify, and distribute it freely.

💡 Future Improvements
🔹 Use more features like location, number of bathrooms, and year built
🔹 Implement Polynomial Regression for better accuracy
🔹 Deploy as a web app using Flask

📢 Contributing
Feel free to fork the repository, create a new branch, and submit a pull request!
