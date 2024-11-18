# 导入所需的库
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成样本数据
# 这里我们使用make_blobs生成一组模拟数据
# n_samples: 数据点数量
# centers: 生成几个簇（这里为3个簇）
# cluster_std: 每个簇的标准差，数值越大，簇越分散
# random_state: 保证生成的随机数据可重复
X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# 创建KMeans模型
# n_clusters: 设定要生成的簇的数量
# init: 初始化方法，'k-means++'可以加速收敛
# max_iter: 最大迭代次数
# n_init: 初始化KMeans的次数，选择最好的一次
# random_state: 保证结果的可重复性
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)

# 用生成的数据拟合KMeans模型
kmeans.fit(X)

# 获取每个数据点对应的簇标签
# labels_属性表示模型预测的每个数据点的簇标号
y_kmeans = kmeans.labels_

# 绘制聚类结果
# 绘制所有数据点，并根据簇标签对点进行着色
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# 获取簇中心
# cluster_centers_属性表示聚类中心的坐标
centers = kmeans.cluster_centers_

# 绘制簇中心点
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)

# 显示结果
plt.show()