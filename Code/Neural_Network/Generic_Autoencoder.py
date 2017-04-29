
# coding: utf-8

# In[0]:

#get_ipython().magic('matplotlib inline')

import sys
import os

sys.path.append("../Pre_processing")

from Data_Preparation_Library import *

sys.path.append(pre_processing_path)

selectedBatches=["6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"]
#maxBatchId = 
#selectedBatches=[str(i) for i in range(maxBatchId)]
batch_data_object = []
for i in selectedBatches:
	with open(os.path.join(temp_path,'full_data_object_' + i + '.p'), 'rb') as handle:
	    batch_data_object+=pickle.load(handle)


# In[37]:

def get_g_Data_for_Autoencoder(data_object):
    X_list=[(data_object[index].g_image_resized_reshaped)/data_object[index].g_image_resized_reshaped.max() for index in range(len(data_object))]
    return np.asarray(X_list),np.asarray(X_list)


# In[38]:

X,y=get_g_Data_for_Autoencoder(batch_data_object)


# In[53]:

# Keras model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

input_img = Input(shape=(128, 128, 1))  # adapt this if using `channels_first` image data format

ws = 3
mpw = 2

x = Conv2D(32, (ws,ws), activation='relu', padding='same')(input_img)
print("1st Convolutional layer shape",x.shape)
x = MaxPooling2D((mpw,mpw), padding='same')(x)
print("1st Maxpooling layer shape",x.shape)
x = Conv2D(16, (ws,ws), activation='relu', padding='same')(x)
print("2nd Convolutional layer shape",x.shape)
x = MaxPooling2D((mpw,mpw), padding='same')(x)
print("2nd Maxpooling layer shape",x.shape)
#x = Conv2D(16, (ws,ws), activation='relu', padding='same')(x)
#print("3rd Convolutional layer shape",x.shape)
#encoded = MaxPooling2D((mpw,mpw), padding='same')(x)
#print("Encoding layer shape",x.shape)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

#x = Conv2D(16, (ws,ws), activation='relu', padding='same')(encoded)
#x = UpSampling2D((mpw,mpw))(x)
x = Conv2D(16, (ws,ws), activation='relu', padding='same')(x)
x = UpSampling2D((mpw,mpw))(x)
x = Conv2D(32, (ws,ws), activation='relu', padding='same')(x)
x = UpSampling2D((mpw,mpw))(x)

decoded = Conv2D(1, (ws,ws), activation='sigmoid', padding='same')(x)

encoder = Model(input_img, encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# In[54]:

X = X.astype('float32')
y = y.astype('float32') 


X_train=X[0:int(len(X)*2/3)]
X_test=X[int(len(X)*2/3):]

# In[56]:

autoencoder.fit(X_train, X_train,
                epochs=30,
                batch_size=10,
                shuffle=True,
                validation_data=(X_test, X_test))


# In[57]:

decoded_imgs_test = autoencoder.predict(X_test)
encoded_imgs_test = encoder.predict(X_test)
decoded_imgs_train = autoencoder.predict(X_train)
encoded_imgs_train = encoder.predict(X_train)


### Save results

#with open(os.path.join(output_path,"autoencoder_results_v3_up_to_" + str(maxBatchId) ), 'wb') as handle:
with open(os.path.join(output_path,"autoencoder_results_test_v1_" + "_".join(selectedBatches) ), 'wb') as handle:
  pickle.dump(decoded_imgs_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(output_path,"autoencoder_results_train_v1_" + "_".join(selectedBatches) ), 'wb') as handle:
  pickle.dump(decoded_imgs_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(output_path,"encoder_results_test_v1_" + "_".join(selectedBatches) ), 'wb') as handle:
  pickle.dump(encoded_imgs_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(output_path,"encoder_results_train_v1_" + "_".join(selectedBatches) ), 'wb') as handle:
  pickle.dump(encoded_imgs_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

