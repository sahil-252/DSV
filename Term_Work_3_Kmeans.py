import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Dataset/Kmeans.csv")

# Convert Gender to numeric
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

# Select features
X = df[['Age', 'Annual_Income', 'Spending_Score']]

# Scale features for better clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit KMeans with chosen clusters (e.g., 4)
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(X_scaled)

# Assign clusters
df['Cluster'] = kmeans.labels_

# Plot Age vs Spending Score colored by cluster
plt.scatter(df['Age'], df['Spending_Score'], c=df['Cluster'], cmap='viridis')
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.title("K-Means Clustering (Age vs Spending Score)")
plt.colorbar(label='Cluster')
plt.show()
