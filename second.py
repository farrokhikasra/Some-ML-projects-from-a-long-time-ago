import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, BatchNormalization
from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np

# np.set_printoptions(precision=7)

data = mnist.load_data()
(train_x, train_y), (test_x, test_y) = data

# plt.imshow(train_x[1])
# plt.show()
# print(train_y[1])

model = Sequential([
    Flatten(input_shape=[28, 28]),
    BatchNormalization(), # deep difference?
    Dense(200, activation='tanh'),
    BatchNormalization(),
    Dense(200, activation='relu'),
    BatchNormalization(),
    Dense(100, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_x, train_y, epochs=3)
model.evaluate(test_x, test_y)

#preds = model.predict(x)
#print('Predicted:', decode_predictions(preds, top=3)[0])