
# coding: utf-8

# In[0]:

#get_ipython().magic('matplotlib inline')

import sys
import os

sys.path.append("../Pre_processing")

from Data_Preparation_Library import *

sys.path.append(pre_processing_path)

#selectedBatches=["6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"]
maxBatchId = 76
selectedBatches=[str(i) for i in range(maxBatchId)]
#extra_folder = "128_128"
extra_folder = "32_32"
batch_data_object = []
for i in selectedBatches:
	with open(os.path.join(temp_path,extra_folder,'full_data_object_' + i + '.p'), 'rb') as handle:
	    batch_data_object+=pickle.load(handle)


# In[37]:

def get_g_Data_for_Autoencoder(data_object):
    X_list = []
    Y_list = []
    for index in range(len(data_object)):
      if data_object[index].i_image != None: 
	      X_list.append((data_object[index].g_image_resized_reshaped)/data_object[index].g_image_resized_reshaped.max())
	      Y_list.append((data_object[index].i_image_resized_reshaped)/data_object[index].i_image_resized_reshaped.max())
    return np.asarray(X_list),np.asarray(Y_list)


# In[38]:

X,Y=get_g_Data_for_Autoencoder(batch_data_object)


# In[53]:

# insert model here  TODO
version = "v1"
from topology_v1 import *

encoder = Model(input_img, encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# In[54]:

X = X.astype('float32')
Y = Y.astype('float32') 


X_train = X[0:int(len(X)*2/3)]
X_test  = X[int(len(X)*2/3):]
Y_train = Y[0:int(len(X)*2/3)]
Y_test  = Y[int(len(X)*2/3):]

# In[56]:

autoencoder.fit(X_train, Y_train,
                epochs=30,
                batch_size=10,
                shuffle=True,
                validation_data=(X_test, Y_test))


# In[57]:

decoded_imgs_test = autoencoder.predict(X_test)
encoded_imgs_test = encoder.predict(X_test)
decoded_imgs_train = autoencoder.predict(X_train)
encoded_imgs_train = encoder.predict(X_train)


### Save results

#postfix = "_".join(selectedBatches)
postfix = "_all"

autoencoder.save(os.path.join(output_path,"g_to_i_model_"+version+"_"+postfix))
encoder.save(os.path.join(output_path,"g_to_i_encoder_model_"+version+"_"+postfix))

#with open(os.path.join(output_path,"autoencoder_results_v3_up_to_" + str(maxBatchId) ), 'wb') as handle:
with open(os.path.join(output_path,"g_to_i_transformer_results_test_"+version+"_" + postfix ), 'wb') as handle:
  pickle.dump(decoded_imgs_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(output_path,"g_to_i_transformer_results_train_"+version+"_" + postfix ), 'wb') as handle:
  pickle.dump(decoded_imgs_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(output_path,"g_to_i_encoder_results_test_"+version+"_" + postfix ), 'wb') as handle:
  pickle.dump(encoded_imgs_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(output_path,"g_to_i_encoder_results_train_"+version+"_" + postfix ), 'wb') as handle:
  pickle.dump(encoded_imgs_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

