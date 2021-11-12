import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from skimage.transform import resize

plt.style.use("fivethirtyeight")

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
class_names = ["airplane","car","bird","cat","deer","dog","frog","horse","ship","track"]

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
x_train = x_train/255
x_test = x_test/255

model = keras.Sequential([
    Conv2D(32,(5,5),activation='relu',input_shape=(32,32,3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(32,(5,5),activation='relu'),
    MaxPooling2D (pool_size=(2,2)),
    Flatten(),
    Dense(1000, activation='relu'),
    Dropout (0.5),
    Dense(500, activation='relu'),
    Dropout(0.5),
    Dense(250, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train_one_hot, batch_size=256, epochs=10, validation_split=0.2)

image_path = "images/image1.jpg"
image_read = plt.imread(image_path)
resized_image = resize(image_read,(32,32,3))
prediction = model.predict(np.array([resized_image]))

list_index = [0,1,2,3,4,5,6,7,8,9]
x = prediction

for i in range (10):
  for j in range (10):
    if x[0][list_index[i]] > x[0][list_index[j]]:
      temp = list_index[i]
      list_index[i] = list_index[j]
      list_index[j] = temp

plt.grid(False)
plt.imshow(resized_image)
print (class_names[list_index[0]])
