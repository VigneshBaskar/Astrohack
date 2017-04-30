import sys
import os

sys.path.append("../Pre_processing")

from Data_Preparation_Library import *

# Keras model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

input_img = Input(shape=(image_size, image_size, 1))  # adapt this if using `channels_first` image data format

ws = 3
mpw = 2

x = Conv2D(image_size//4, (ws,ws), activation='relu', padding='same')(input_img)
print("1st Convolutional layer shape",x.shape)
x = MaxPooling2D((mpw,mpw), padding='same')(x)
print("1st Maxpooling layer shape",x.shape)
x = Conv2D(image_size//8, (ws,ws), activation='relu', padding='same')(x)
print("2nd Convolutional layer shape",x.shape)
x = MaxPooling2D((mpw,mpw), padding='same')(x)
print("2nd Maxpooling layer shape",x.shape)
x = Conv2D(image_size//8, (ws,ws), activation='relu', padding='same')(x)
print("3rd Convolutional layer shape",x.shape)
encoded = MaxPooling2D((mpw,mpw), padding='same')(x)
print("Encoding layer shape",x.shape)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(image_size//8, (ws,ws), activation='relu', padding='same')(encoded)
x = UpSampling2D((mpw,mpw))(x)
x = Conv2D(image_size//8, (ws,ws), activation='relu', padding='same')(x)
x = UpSampling2D((mpw,mpw))(x)
x = Conv2D(image_size//4, (ws,ws), activation='relu', padding='same')(x)
x = UpSampling2D((mpw,mpw))(x)

decoded = Conv2D(1, (ws,ws), activation='sigmoid', padding='same')(x)

print("Executing topology v1")
