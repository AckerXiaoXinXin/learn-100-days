import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy


(x_train, x_test), (y_train, y_test) = mnist.load_data()
x_train = x_train/255
x_test = x_test/255

x_train.astype('float32')
x_test.astype('float32')

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='sigmoid'),
    Dense(10, activation='softmax'),
])

model.compile(
    optimizer=Adam(),
    loss=SparseCategoricalCrossentropy(),
    metrics=[SparseTopKCategoricalAccuracy],
)

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=0.2)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'test loss: {test_loss:}, test accuracy: {test_acc:.2f}')


prediction = model.predict(x_test)
print(prediction[0])
