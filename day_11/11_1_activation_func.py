import tensorflow as tf
from tensorflow.keras import models, layers


model = models.Sequential()

model.add(layers.Dense(64, activation='relu', input_shape=(32,)))
model.add(layers.Dense(64, activation='sigmoid'))
model.add(layers.Dense(64, activation='tanh'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()