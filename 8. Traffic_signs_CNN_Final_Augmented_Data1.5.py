!git clone https://bitbucket.org/jadslim/german-traffic-signs

!ls german-traffic-signs

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle
import pandas as pd
import random
np.random.seed(0)

with open('german-traffic-signs/train.p', "rb") as f:
  train_data = pickle.load(f)

with open('german-traffic-signs/valid.p', "rb") as f:
  val_data = pickle.load(f)
  
with open('german-traffic-signs/test.p', "rb") as f:
  test_data = pickle.load(f)
  
  
print(type(train_data))
  

X_train , y_train = train_data['features'], train_data['labels']
X_val , y_val = val_data['features'], val_data['labels']
X_test , y_test = test_data['features'], test_data['labels']


print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels"
assert(X_val.shape[0] == y_val.shape[0]), "The number of images is not equal to the number of labels"
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels"
assert(X_train.shape[1:] == (32, 32, 3)), "The dimensions are not equal to 32x32x3"
assert(X_val.shape[1:] == (32, 32, 3)), "The dimensions are not equal to 32x32x3"
assert(X_test.shape[1:] ==(32, 32, 3)), "The dimensions are not equal to 32x32x3"



data = pd.read_csv('german-traffic-signs/signnames.csv')
print(data)



num_of_samples = []

cols = 5
num_classes = 43

fig, axs = plt.subplots(nrows=num_classes, ncols = cols, figsize=(5, 50))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows(): # "j" will collect index and "row" will collect value.
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected - 1)), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row['SignName'])
            num_of_samples.append(len(x_selected))



plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")

import cv2
plt.imshow(X_train[1000])
plt.axis("off")
print(X_train[1000].shape)
print(y_train[1000])



def grayscale(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return img


img = grayscale(X_train[1000])
plt.axis("off")
plt.imshow(img, cmap="gray")
print(img.shape)

def equalize(img):
  img = cv2.equalizeHist(img)
  return img

img = equalize(img)
plt.imshow(img, cmap='gray')
plt.axis("off")
print(img.shape)


def preprocessing(img):
  img = grayscale(img)
  img = equalize(img)
  img = img/255
  return img

X_train = np.array(list(map(preproceesing, X_train)))
X_val = np.array(list(map(preproceesing, X_val)))
X_test = np.array(list(map(preproceesing, X_test)))

plt.imshow(X_train[random.randint(0, len(X_train) -1)], cmap = 'gray')
plt.axis('off')
print(X_train.shape)

X_train = X_train.reshape(34799, 32, 32, 1)
X_val = X_val.reshape(4410,32, 32, 1)
X_test = X_test.reshape(12630, 32,32, 1)


#Data Augmentataion

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(width_shift_range=0.1,
                  height_shift_range=0.1,
                  zoom_range=0.2,
                  shear_range=0.1,
                  rotation_range=10)

datagen.fit(X_train)

batches = datagen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

fig, axs = plt.subplots(1, 15, figsize = (20,5))
fig.tight_layout()

for i in range(15):
  axs[i].imshow(X_batch[i].reshape(32, 32), cmap = 'gray')
  axs[i].axis('off')

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)
y_test = to_categorical(y_test, 43)

def leNet_model():
  model = Sequential()
  model.add(Conv2D(60, (5, 5), input_shape= (32, 32, 1), activation='relu'))
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(Conv2D(30, (3, 3), activation = 'relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Flatten())
  model.add(Dense(500, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))
  #Compile model
  model.compile(Adam(lr=0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
  return model

def modified_leNet_model():
  model = Sequential()
  model.add(Conv2D(60, (5, 5), input_shape= (32, 32, 1), activation='relu'))
  model.add(Conv2D(60, (5, 5), activation='relu'))
  model.add(MaxPooling2D(pool_size = (2,2)))
  
  model.add(Conv2D(30, (3, 3), activation = 'relu'))
  model.add(Conv2D(30, (3, 3), activation = 'relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
#   model.add(Dropout(0.5))
  
  model.add(Flatten())
  model.add(Dense(500, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))
  #Compile model
  model.compile(Adam(lr=0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
  return model

model = modified_leNet_model()
print(model.summary())

model1 = leNet_model()
print(model1.summary())

h = model.fit_generator(datagen.flow(X_train, y_train, batch_size = 50), steps_per_epoch=2000, epochs = 10, validation_data=(X_val, y_val), shuffle= 1)
# h = model.fit(X_train, y_train, epochs = 10, validation_data=(X_val, y_val), batch_size= 400, verbose = 1, shuffle = 1)

plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['training', 'validation'])
plt.title("Accuracy")
plt.xlabel("epoch")

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title("Loss")
plt.xlabel("epoch")

score = model.evaluate(X_test,y_test, verbose = 0)
print("Test score",  score[0])
print("Test accuracy", score[1])

#Testing our data by importing pictures from web and preproccessing it
#fetch image

import requests
from PIL import Image
url_1 = 'https://c8.alamy.com/comp/G667W0/road-sign-speed-limit-30-kmh-zone-passau-bavaria-germany-G667W0.jpg'
url_2 = 'https://c8.alamy.com/comp/A0RX23/cars-and-automobiles-must-turn-left-ahead-sign-A0RX23.jpg'
url_3 = 'https://c8.alamy.com/comp/J2MRAJ/german-road-sign-bicycles-crossing-J2MRAJ.jpg'
url_4 = 'https://previews.123rf.com/images/pejo/pejo0907/pejo090700003/5155701-german-traffic-sign-no-205-give-way.jpg'
url_5 = 'https://previews.123rf.com/images/bwylezich/bwylezich1608/bwylezich160800375/64914157-german-road-sign-slippery-road.jpg'


r = requests.get(url_4, stream=True)
img = Image.open(r.raw)
plt.imshow(img, cmap=plt.get_cmap('gray'))


#Preprocess image

img = np.asarray(img)
img = cv2.resize(img, (32, 32))
img = preprocessing(img)
plt.imshow(img, cmap = plt.get_cmap('gray'))
print(img.shape)

#Reshape reshape

img = img.reshape(1, 32, 32, 1)

#Test image
print("predicted sign: "+ str(model.predict_classes(img)))




x_sel = X_train[y_train == 30]
plt.imshow(x_sel[random.randint(0, len(x_sel - 1)), :, :], cmap=plt.get_cmap("gray"))

print(X_train.shape)
print(y_train.shape)

