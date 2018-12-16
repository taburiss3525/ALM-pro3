from keras.datasets import mnist
from keras.layers import MaxPooling2D
#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt
#plot the first image in the dataset
plt.imshow(X_train[0],cmap='gray')

X_train[0].shape

X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

from keras.utils import to_categorical
#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train[0]

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, (5,5), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

model.predict(X_test[:4])
y_test[:4]

a = model.get_weights()
firstlayer = []
for i in range(64):
    firstlayer.append(a[0][:,:,0,i])
    
secondlayer = []
for i in range(32):
    secondlayer.append(a[2][:,:,0,i])
    
l1f1 = a[0][:,:,0,0]

plt.imshow(a[0][:,:,0,1])

for Filter in firstlayer:
    l1f1_in_tmp = []
    for row in Filter:
        row_tmp = []
        for entry in row:
            if entry >=0:
                row_tmp.append(1)
            else:
                row_tmp.append(0)
        l1f1_in_tmp.append(row_tmp)
    plt.imshow(l1f1_in_tmp, cmap = 'gray')
    plt.show()

for Filter in secondlayer:
    l1f1_in_tmp = []
    for row in Filter:
        row_tmp = []
        for entry in row:
            if entry >=0:
                row_tmp.append(1)
            else:
                row_tmp.append(0)
        l1f1_in_tmp.append(row_tmp)
    plt.imshow(l1f1_in_tmp, cmap = 'gray')
    plt.show()


test = [[12,1,1],
        [1,0,0],
        [0,1,0]]

plt.imshow(test, cmap = 'gray')


