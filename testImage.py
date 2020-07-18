from keras.preprocessing.image import img_to_array
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.utils import to_categorical
from keras import layers
from keras import models
from PIL import Image
import PIL
import cv2

from keras.models import model_from_json


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

fname = input("Enter the image filename to estimate:\n")
image = cv2.imread(fname)
output = image.copy()

# pre-process the image for classification
#image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
img_gray = np.expand_dims(np.mean(image, axis=2), 2)
img_test = np.expand_dims(img_gray, axis=0)
print('Grayscale image shape:', img_gray.shape)

pred = loaded_model.predict(img_test)

pred_pos = np.argmax(pred)

if pred_pos==0:
    print("I think this is a T-shirt or top")
elif pred_pos==1:
    print("I think this is a pair of pants or jeans")
elif pred_pos == 2:
    print("I think this is a sweater/pullover")
elif pred_pos==3:
    print("I think this is a dress")
elif pred_pos==4:
    print("I think this is a coat")
elif pred_pos==5:
    print("I think this is a pair of sandals...or odd looking shoe.")
elif pred_pos==6:
    print("I think this is a shirt")
elif pred_pos==7:
    print("I think this is a sneaker.")
elif pred_pos==8:
    print("I think this is a bag.")
elif pred_pos == 9:
    print("I think this is an ankle boot")