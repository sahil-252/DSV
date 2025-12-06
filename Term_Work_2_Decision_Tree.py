import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("Dataset/Decision_Tree.csv")

# Convert categorical Rain to 0/1 (if needed)
df['Rain'] = df['Rain'].map({'rain': 1, 'no rain': 0})

# Features (X) and Target (y)
X = df[['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']]
y = df['Rain']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create Decision Tree Model
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Example Prediction
sample = pd.DataFrame({
    'Temperature': [15],
    'Humidity': [80],
    'Wind_Speed': [12],
    'Cloud_Cover': [60],
    'Pressure': [1010]
})

prediction = model.predict(sample)[0]

print("\nPrediction:", "Rain" if prediction == 1 else "No Rain")
