# pca:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


iris = load_iris()

X, y = iris.data, iris.target

Scaler = StandardScaler()
X_Scaled = Scaler.fit_transform(X)

pca = PCA(n_components=2)
pca.fit_transform(X_Scaled)

e_v = pca.explained_variance_ratio_
print(f'explained_variance: {e_v}')

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=100)
plt.title('pca')
plt.xlabel('x')
plt.ylabel('y')

legend = plt.legend(*scatter.legend_elements(), title='Classes')
plt.show()





