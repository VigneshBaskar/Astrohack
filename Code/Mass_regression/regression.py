import sys
import os

sys.path.append("../Pre_processing")

from Data_Preparation_Library import *

#selectedBatches=["6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"]
maxBatchId = 1
selectedBatches=[str(i) for i in range(maxBatchId)]

batch_data_object = []
for i in selectedBatches:
	with open(os.path.join(temp_path,'full_data_object_' + i + '.p'), 'rb') as handle:
	    batch_data_object+=pickle.load(handle)

data_train = batch_data_object[0:int(len(X)*2/3)]
#data_test  = batch_data_object[int(len(X)*2/3):]

###################
#
# features
#

#def pca_feature(d):
#  return [v1_i,v2_i,v1_i/v2_i,v2_i/v1_i, v1_g,v2_g,v1_g/v2_g,v2_g/v1_g]

def flux(d):
#
  # TODO: get flux_i from model
  if d.i_image != None:
	  flux_i = sum(sum(d.i_image))
  else:
    flux_i = 0
#
  flux_g = sum(sum(d.g_image))
  return flux_i+flux_g, flux_g-flux_i

# accessible in the class: size of image, maxium of image, normalization factor of g + fitted value of normalization factor for i

###################
#
# encrich encoder
#

with open(os.path.join(output_path,"encoder_results_train_v2_" + "_".join(selectedBatches)),'rb') as handle:
  encoded_imgs=pickle.load(handle)

encoded_imgs_reshaped = [e.reshape(-1,1) for e in encoded_imgs]

###################
#
# merge encoder features with other features
#

features = []
for i in range(len(data_train)):
  e = encoded_imgs_reshaped[i]
  d = data_train[i]
#  
  # flux
  f1,f2 = flux(d)
#
  #global shape features
#  slist = pca_feature(d)
  # org image size
  os=d.g_image.shape[0]
#
  features.append( np.append(e,np.array([f1,f2,os]))) # + slist


###################
#
# regression
#


