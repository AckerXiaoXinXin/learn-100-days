from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys

sys.setrecursionlimit(5000)


X, y = make_blobs(n_samples=300, centers=5, cluster_std=0.60, random_state=42)

print(type(X))
print(X.shape)
# Scaler = StandardScaler()
pca = PCA(n_components=2)

X = pca.fit_transform(X)

kmeans = MiniBatchKMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=20, random_state=42)

kmeans.fit(X)

y_kmeans = kmeans.labels_

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)

plt.show()




