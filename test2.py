import numpy as np
import torch as t
from torch.nn.functional import softmax
import keras


def read_labels(filename, items):
    file_labels = open(filename, 'rb')
    file_labels.seek(8)
    data = file_labels.read(items)
    y = np.zeros(items, dtype=np.int64)
    for i in range(items):
        y[i] = data[i]
    file_labels.close()
    return y.reshape(-1,1)


y_train = read_labels('/Users/ziyingliuziying/lusing/mnist/train-labels-idx1-ubyte', 60000)
y_test = read_labels('/Users/ziyingliuziying/lusing/mnist/t10k-labels-idx1-ubyte', 10000)


def read_images(filename, items):
    file_image = open(filename, 'rb')
    file_image.seek(16)

    data = file_image.read(items * 28 * 28)

    X = np.zeros(items * 28 * 28, dtype=np.float32)
    for i in range(items * 28 * 28):
        X[i] = data[i] / 255
    file_image.close()
    return X.reshape(-1, 28 * 28)


X_train = read_images('/Users/ziyingliuziying/lusing/mnist/train-images-idx3-ubyte', 60000)
X_test = read_images('/Users/ziyingliuziying/lusing/mnist/t10k-images-idx3-ubyte', 10000)

#y_train = keras.utils.to_categorical(y_train, 10)
#y_test = keras.utils.to_categorical(y_test, 10)


class SimpleNet(t.nn.Module):
    def __init__(self, input, hidden1, hidden2, output):
        super(SimpleNet, self).__init__()
        self.layer1 = t.nn.Sequential(t.nn.Linear(input, hidden1), t.nn.ReLU(True))
        self.layer2 = t.nn.Sequential(t.nn.Linear(hidden1, hidden2), t.nn.ReLU(True))
        self.layer3 = t.nn.Sequential(t.nn.Linear(hidden2, output), t.nn.ReLU(True))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return t.nn.functional.softmax(x,dim=1)


model = SimpleNet(28 * 28, 300, 100, 10)

criterion = t.nn.CrossEntropyLoss()
optimizer = t.optim.SGD(model.parameters(), lr=1e-3)

num_epochs = 1000

for epoch in range(num_epochs):
    X = t.autograd.Variable(t.from_numpy(X_train))
    y = t.autograd.Variable(t.from_numpy(y_train))

    # print(X)

    # 正向传播

    out = model(X)
    print(out.shape)
    print(y.shape)
    loss = criterion(out, y)

    # 反向梯度下降

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss.item())

model.eval()
eval_loss = 0
eval_acc = 0

# model = Sequential()
# model.add(Conv2D(128, kernel_size=(3, 3),
#                 activation='relu',
#                 input_shape=(28,28,1)))
# model.add(Conv2D(64, (5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(10, activation='softmax'))

# model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adadelta(),
#              metrics=['accuracy'])

# model.fit(X_train, y_train,
#          batch_size=128,
#          epochs=10,
#          verbose=1,
#          validation_data=(X_test, y_test))
# score = model.evaluate(X_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])