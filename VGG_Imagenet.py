import numpy as np
import os
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator
import h5py
from keras.models import load_model
import cv2
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
model = load_model('saved_models/cifar-10.h5')
# Compile the model
model.compile(loss='categorical_crossentropy',
        optimizer=Adam(lr=0.0001, decay=1e-6),
        metrics=['accuracy'])

from keras.preprocessing import image

input_img=[]
os.chdir('video_dataset/Frame')
for f in os.listdir():
    img = image.load_img(f, target_size=(32, 32))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    input_img.append(x)
    
    
images=np.vstack(input_img)
classes = model.predict_classes(images, batch_size=10)

print('The Prediction for the images are :')
pred = []
for response in classes:
    if response == 1 or response == 9:
        pred.append('CAR')
    else:
        pred.append('NON-CAR')

count = 0;
font = cv2.FONT_HERSHEY_SIMPLEX
for f in os.listdir():
    img = cv2.imread(f)
    img = cv2.resize(img,(640,320))
    cv2.putText(img,pred[count], (250,60), font, 1.5, (0, 0, 255),2)
    cv2.imshow('Play',img)
    cv2.waitKey(500)
    count = count + 1
cv2.destroyAllWindows()

