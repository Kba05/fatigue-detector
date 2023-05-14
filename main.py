import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.preprocessing import OneHotEncoder
import dlib
from keras.applications.resnet import ResNet50
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.utils.vis_utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,GlobalAveragePooling2D,Concatenate, ReLU, LeakyReLU,Reshape, Lambda
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import metrics
from skimage.transform import resize
from tensorflow.python.client import device_lib

detector = dlib.get_frontal_face_detector()
val_base = 'E:/input2/val'
train_base = 'E:/input2/train'
model = load_model("fatigue_model4.model")
classes_name = ['бодрый','устал - иди отдохни, или скажи мяу']


def predict_image(path, model):
        imageOrig = cv2.imread(path)
        cv2.cvtColor(imageOrig, cv2.COLOR_BGR2RGB)
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) > 0:
            tl_x = faces[0].tl_corner().x
            tl_y = faces[0].tl_corner().y
            tl_h = faces[0].height()
            tl_w = faces[0].width()
            if tl_x > 0 and tl_y > 0 and tl_h > 10 and tl_w > 10:
                image = image[tl_y : tl_y + tl_w, tl_x:tl_x + tl_h, :]
        image_norm = image/255.0
        im = cv2.resize(image_norm, (48, 48))
        prediction = model.predict(im[np.newaxis])
        index = np.argmax(prediction)
        print(str(classes_name[index]))
        #plt.title("{}".format(str(classes_name[index]).title()), size=18, color='red')
        #plt.imshow(imageOrig[:, :, ::-1])


predict_image("E:/input2/val/active/image_0393.jpg", model)