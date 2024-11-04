import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# 数据预处理：将数据从 (28, 28) 格式转换为 (28, 28, 1)，并归一化到 [0, 1] 范围
x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255
x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255

# 输出类别数（0-9）
num_classes = 10
# 对标签进行 one-hot 编码
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# 创建 CNN 模型
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),  # 展平特征图
    layers.Dense(128, activation='relu'),  # 全连接层
    layers.Dense(num_classes, activation='softmax')  # 输出层
])

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 打印模型概述
model.summary()

# 训练模型
model.fit(x_train, y_train, epochs=2, batch_size=64, validation_split=0.2)

# 评估模型
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")