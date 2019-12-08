import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import keras.utils

matplotlib.use('TkAgg')

# Data loading
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Data preparation
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

class_number = 10

## One hot encoding
Y_train = keras.utils.np_utils.to_categorical(y_train, class_number)
Y_test = keras.utils.np_utils.to_categorical(y_test, class_number)

# Model creation
model = Sequential()
model.add(Dense(1024, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(1024))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Model training
model.fit(X_train, Y_train, epochs=30, verbose=2,  validation_data=(X_test, Y_test))

# Model evaluation
results = model.evaluate(X_test,  Y_test, verbose=2)

print("loss", results[0])
print("accuracy", results[1])

# Predictions
predicted_classes = model.predict_classes(X_test)
correct_cases = np.nonzero(predicted_classes == y_test)[0]
wrong_cases = np.nonzero(predicted_classes != y_test)[0]
print()
print(len(correct_cases), " correct predictions")
print(len(wrong_cases), " wrong predictions")

plt.rcParams['figure.figsize'] = (4, 8)

for i, correct in enumerate(correct_cases[:3]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[correct].reshape(28, 28))
    plt.title(
        "X: {}, Y: {}".format(predicted_classes[correct],
                                          y_test[correct]))

for i, incorrect in enumerate(wrong_cases[:6]):
    plt.subplot(3, 3, i + 4)
    plt.imshow(X_test[incorrect].reshape(28, 28))
    plt.title(
        "X: {}, Y: {}".format(predicted_classes[incorrect],
                                         y_test[incorrect]))

plt.show()
