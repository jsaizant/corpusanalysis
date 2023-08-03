# Import libraries
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load data
categories = ["alt.atheism", "talk.religion.misc", "comp.graphics", "sci.space"]
dataset = fetch_20newsgroups(
    remove=("headers", "footers", "quotes"),
    subset="all",
    categories=categories,
    shuffle=True,
    random_state=42,
)
documents = dataset.data
labels = dataset.target

# Vectorize documents into sentence embeddings
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(documents)

# Reduce dimensionality with PCA
pca = PCA(n_components=2, random_state=42)
X_reduced = pca.fit_transform(X.toarray())

print(X_reduced)
print(X_reduced.shape)

# Cluster documents with K-means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_reduced)
clusters = kmeans.labels_

# Print the cluster assignments and the cluster centers
print("Cluster assignments:", clusters)
print("Cluster centers:", kmeans.cluster_centers_)


# Import matplotlib
import matplotlib.pyplot as plt

# Plot the reduced data points
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap="viridis")

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="black", s=200, alpha=0.5)

# Show the plot
plt.show()
