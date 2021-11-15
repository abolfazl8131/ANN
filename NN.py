from keras.datasets import cifar10 as cf10
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from pathlib import Path
import  tensorflow
#load dataset
(x_train, y_train), (x_test, y_test) = cf10.load_data()

#normalize

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train = x_train/255
x_test = x_test/255

#set them to matrix
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# model
model = Sequential()

model.add(Conv2D(32,(3,3), activation="relu", padding="same", input_shape=(32,32,3)))
model.add(Conv2D(32,(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(64,(3,3), activation="relu", padding="same"))
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.50))


model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dense(10,activation='softmax'))

#compile
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train,y_train, batch_size=32, epochs=30, validation_data=(x_test, y_test), shuffle=True)

#save the NN structue
model_structure = model.to_json()
f = Path("model_structure.json")
f.write_text(model_structure)

#save NN weights
model.save_weights("model_weights.h5")