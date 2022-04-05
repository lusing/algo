import numpy as np
import keras


def read_labels(filename, items):
    file_labels = open(filename, 'rb')
    file_labels.seek(8)
    data = file_labels.read(items)
    y = np.zeros(items)
    for i in range(items):
        y[i] = data[i]
    file_labels.close()
    return y


y_train = read_labels('./train-labels-idx1-ubyte', 60000)
y_test = read_labels('./t10k-labels-idx1-ubyte', 10000)


def read_images(filename, items):
    file_image = open(filename, 'rb')
    file_image.seek(16)

    data = file_image.read(items*28*28)

    X = np.zeros(items*28*28)
    for i in range(items*28*28):
        X[i] = data[i]/255
    file_image.close()
    return X.reshape(-1,28*28)


X_train = read_images('train-images-idx3-ubyte', 60000)
X_test = read_images('./t10k-images-idx3-ubyte', 10000)


y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(y_train)
print(y_test)

from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()

model.add(Dense(units=121, input_dim=28*28))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])


model.fit(X_train, y_train,
          batch_size=64,
          epochs=20,
          verbose=1,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
