import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# 载入MNIST数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# 构建模型
def build_model(l2_lambda=0.0, dropout_rate=0.0):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))  # 把28x28图像展平
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(l2_lambda)))  # L2正则化
    model.add(Dropout(dropout_rate))  # Dropout层
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(l2_lambda)))  # L2正则化
    model.add(Dropout(dropout_rate))  # Dropout层
    model.add(Dense(10, activation='softmax'))
    return model

# 超参数
l2_lambda = 0.001  # L2 正则化强度
dropout_rate = 0.5  # Dropout 比率

# 用于调整超参数的模型
model = build_model(l2_lambda=l2_lambda, dropout_rate=dropout_rate)

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=128)

# # 绘制训练和验证准确率
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()
#
# # 绘制训练和验证损失
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()

# 测试模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'test loss: {test_loss:.4f}')
print(f'test accuracy: {test_acc:.4f}')