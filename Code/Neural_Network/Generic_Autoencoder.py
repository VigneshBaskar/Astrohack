
# coding: utf-8

# In[38]:

#get_ipython().magic('matplotlib inline')

import os
import sys

try:
    current_path=os.path.dirname(os.path.realpath(__file__))
except NameError:
    current_path=os.getcwd()

for i in range(3):
    sys.path.append(current_path)
    current_path=os.path.dirname(current_path)

from param_global import *
sys.path.append(pre_processing_path)
from Read_Preprocess_data import data # ,plot_images
import random


# In[2]:

with open(os.path.join(sample_data_path,'sample_data_object.p'), 'rb') as handle:
    sample_data_object=pickle.load(handle)


# In[7]:

def get_g_Data_for_Autoencoder(data_object):
    X_list=[data_object[i].g_image_resized_reshaped for index in range(len(data_object))]
    return np.asarray(X_list),np.asarray(X_list)


# In[8]:

X,y=get_g_Data_for_Autoencoder(sample_data_object)


# In[60]:

# Keras model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

input_img = Input(shape=(128, 128, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')


# In[61]:

X = X.astype('float32')
y = y.astype('float32') 


# In[62]:

X_train=X[0:int(len(X)*2/3)]
X_test=X[int(len(X)*2/3):]


# In[76]:

#X_train=X


# In[77]:

autoencoder.fit(X_train, X_train,
                epochs=30,
                batch_size=10,
                shuffle=True,
                validation_data=(X_test, X_test))


# In[78]:

decoded_imgs = autoencoder.predict(X_train)


### Save results

with open(os.path.join(output_path,"autoencoder_on_sample"), 'wb') as handle:
  pickle.dump(decoded_imgs, handle, protocol=pickle.HIGHEST_PROTOCOL)

#with open(os.path.join(output_path,"autoencoder_on_sample"), 'rb') as handle:
#  b = pickle.load(handle)

# In[87]:

#plot_images([decoded_imgs[20].reshape(decoded_imgs[1].shape[0],decoded_imgs[1].shape[1])])


# In[80]:

#sample_data_object[9].plot_image()


# In[ ]:



