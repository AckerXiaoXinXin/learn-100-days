import numpy as np
import matplotlib.pyplot as plt


# 折线图
# x = np.linspace(1, 10, 100)
# y = np.sin(x)
#
# plt.plot(x, y, label="Sine Wave", color='b', linestyle='-')
# plt.title("Sine Wave Function")
# plt.xlabel("X")
# plt.ylabel("Y")
#
# plt.legend()
# plt.show()


# 散点图
# np.random.seed(0)
# x = np.random.rand(50)
# y = np.random.rand(50)
#
# plt.scatter(x, y, color='r', alpha=0.5)
#
# plt.title('Scatter Plot')
# plt.xlabel('X')
# plt.ylabel('Y')
#
# plt.show()


cate = [9, 5, 2, 7]
value = [9, 5, 2, 7]

x = np.arange(len(cate))
plt.bar(x, value, color='g')

plt.xticks(x, labels=cate)


plt.show()