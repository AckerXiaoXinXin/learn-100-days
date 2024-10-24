import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 手动创建数据集
# 数据集由三组二维数据点组成
X = np.array([
    [1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11],
    [8, 2], [10, 2], [9, 3], [4, 7], [2, 4], [6, 8]
])

# 创建KMeans模型
# 我们这里设置聚类数量为3
kmeans = KMeans(n_clusters=3, max_iter=300, n_init=10, random_state=0)

# 训练模型
kmeans.fit(X)

# 获取每个数据点的簇标签
y_kmeans = kmeans.labels_

# 获取簇中心
centers = kmeans.cluster_centers_

# 绘制结果
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# 绘制簇中心点
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)

plt.show()