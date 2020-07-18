from keras.preprocessing.image import img_to_array
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.utils import to_categorical
from keras import layers
from keras import models
from PIL import Image
import PIL

train_df = pd.read_csv('fashion-mnist_train.csv')
test_df = pd.read_csv('fashion-mnist_test.csv')

train_X = train_df.iloc[:,1:len(train_df.columns)]
train_y = train_df[['label']]
train_X = np.asarray(train_X)
train_X = train_X.reshape((60000,28,28,1))
train_X = train_X.astype('float32') / 255

test_X = test_df.iloc[:,1:len(test_df.columns)]
test_y = test_df[['label']]
test_X = np.asarray(test_X)
test_X = test_X.reshape((10000,28,28,1))
test_X = test_X.astype('float32') / 255

train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_X, train_y, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_X, test_y)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")