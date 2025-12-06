import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load CSV
df = pd.read_csv("Dataset/Logistic_Regression.csv")

# Convert "Rain" / "No Rain" into 1 / 0
df['Rain'] = df['Rain'].map({'rain': 1, 'no rain': 0})

# Features and target
X = df[['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']]
y = df['Rain']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Example prediction
sample = pd.DataFrame({
    'Temperature': [15],
    'Humidity': [90],
    'Wind_Speed': [10],
    'Cloud_Cover': [69],
    'Pressure': [1100]
})

prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

print("\nPrediction:", "Rain" if prediction == 1 else "No Rain")
print("Probability of Rain:", round(probability, 2))
