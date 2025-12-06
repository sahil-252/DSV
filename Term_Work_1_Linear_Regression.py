import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("Dataset/Linear_Regression.csv")

# Features and Target
X = df[['YearsExperience']]
y = df['Salary']

# Model
model = LinearRegression()
model.fit(X, y)

# Print slope & intercept
print(f"Slope (m): {model.coef_[0]:.2f}")
print(f"Intercept (c): {model.intercept_:.2f}")

# Predict for 5 years experience
experience = 1
X_test = pd.DataFrame({'YearsExperience': [experience]})
prediction = model.predict(X_test)

print(f"Predicted Salary for {experience} years experience: {prediction[0]:.2f}")

# ============================
#   Linear Regression Graph
# ============================

# Predicted values for line
y_pred = model.predict(X)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Actual Data")       # scatter points
plt.plot(X, y_pred, label="Regression Line") # regression line
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression - Salary Prediction")
plt.legend()
plt.show()
