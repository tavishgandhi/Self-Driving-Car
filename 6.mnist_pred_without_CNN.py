import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random

np.random.seed(0)

(X_train, y_train), (X_test, y_test) = mnist.load_data()


print(X_train.shape)
print(X_test.shape)

assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to number of labels."
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to number of labels."
assert(X_train.shape[1:] == (28,28)), "The dimensions of images is not 28 x 28."
assert(X_test.shape[1:] == (28,28)), "The dimensions of images is not 28 x 28."


num_of_sample = []
cols = 5
num_classes = 10
fig, axs = plt.subplots(nrows = num_classes, ncols = cols, figsize=(5,10))
fig.tight_layout()
for i in range(cols):
    for j in range(num_classes):
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected - 1)), :, :] , cmap = plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(j)  
            num_of_sample.append(len(x_selected))

        

print(num_of_sample)
plt.figure(figsize = (12,4))
plt.bar(range(0, num_classes), num_of_sample)
plt.title("Distribution of training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of Images")

# One hot Encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Feature selection
X_train = X_train/255
X_test = X_test/255

num_pixels = 784
X_train = X_train.reshape(X_train.shape[0], num_pixels)
print(X_train.shape)
X_test = X_test.reshape(X_test.shape[0], num_pixels)
print(X_test.shape)

def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim = num_pixels, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(num_classes, activation= 'softmax'))
    model.compile(Adam(lr=0.01), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

model = create_model()
print(model.summary())

h =model.fit(X_train, y_train, validation_split = 0.1, epochs = 10, batch_size = 200, verbose = 1, shuffle = 1)

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['loss' , 'val_loss'])
plt.title("loss")
plt.xlabel("epoch")

plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['acc' , 'val_acc'])
plt.title("Accuracy")
plt.xlabel("epoch")

score = model.evaluate(X_test, y_test, verbose = 0)
print("Test Score : ", score[0])
print("Test Accuracy : ", score[1])

import requests
from PIL import Image
url = 'https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png'
response = requests.get(url, stream = True)
img = Image.open(response.raw)
plt.imshow(img)


img_array = np.asarray(img)

print(img_array.shape)

import cv2
resized = cv2.resize(img_array, (28, 28))
gray_scale = cv2.cvtColor(resized , cv2.COLOR_BGR2GRAY)
plt.imshow(image, cmap= plt.get_cmap("gray"))
image = cv2.bitwise_not(gray_scale)

image = image/255
image = image.reshape(1, 784)

print(image)

prediction = model.predict_classes(image)

print(prediction)

