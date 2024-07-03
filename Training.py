import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.models import load_model, model_from_json
from keras.layers import Conv2D,Activation,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization

# import torch.nn as nn

x_train=np.load("train_x.npy")
y_train=np.load("train_y.npy")
x_test=np.load("test_x.npy")
y_test=np.load("test_y.npy")
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu"))
model.add(BatchNormalization())

model.add(Conv2D(32, (5, 5), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Flatten())

model.add(Dense(256, activation="relu"))
model.add(Dense(36, activation="softmax"))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(
    x_train,
    y_train,
    validation_data=(x_test,y_test), 
    epochs=5, 
    batch_size=200, 
)
model.save('my_model.keras')
model.save('model.h5')
keras.saving.save_model(model, 'my_model_save2.keras')





