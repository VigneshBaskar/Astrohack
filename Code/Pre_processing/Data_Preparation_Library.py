
# coding: utf-8

# In[22]:

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


# In[23]:

#The word image and img usually refers to np.ndarray


# In[24]:

def image_resize(image,dim = (128, 128)):
    #Args: image in the form of np.ndarray and the dimension to which it has to be resized
    image_resized=cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)
    return image_resized


# In[25]:

def image_2d_to_3d(image):
    #Args: Pass in a 2d dimensional image and get back the same image as 3d
    reshaped_image=image.reshape(image.shape[0],image.shape[1],1)
    return reshaped_image
    


# In[26]:

def plot_images(images):
    #Args: List of images to be plotted
    for i,img in enumerate(images):
        plt.subplot(1,len(images),i+1)
        img=plt.imshow(img)
        img.set_cmap('hot')
        plt.axis("off")


# In[27]:

class data(object):
    #An object with SDSS_ID, logMstar, err_logMstar, Distance, image_data_dir,i_image and g_image.
    #It has functions such as plot_image to get the image and the image resized to dims in the argument
#    
    def generate_i_Image(self):
        if os.path.exists(os.path.join(self.image_data_dir,self.SDSS_ID+"-i.csv")):
            image_df=pd.read_csv(os.path.join(self.image_data_dir,self.SDSS_ID+"-i.csv"),header=None)
#            
            return image_df.as_matrix()
        else:
            return None
    def generate_g_Image(self):
        if os.path.exists(os.path.join(self.image_data_dir,self.SDSS_ID+"-g.csv")):
            image_df=pd.read_csv(os.path.join(self.image_data_dir,self.SDSS_ID+"-g.csv"),header=None)
            return image_df.as_matrix()
        else:
            return None
#    
#    
    def __init__(self,SDSS_ID,logMstar,err_logMstar,Distance,image_data_dir):
        self.SDSS_ID=SDSS_ID
        self.logMstar=logMstar
        self.err_logMstar=err_logMstar
        self.Distance=Distance
        self.image_data_dir=image_data_dir
        self.i_image=self.generate_i_Image()
        self.g_image=self.generate_g_Image() 
#        
        if self.i_image!=None:
            self.i_image_resized=image_resize(self.i_image)
        else:
            self.i_image_resized=None
#            
        if self.g_image!=None:
            self.g_image_resized=image_resize(self.g_image)
        else:
            self.g_image_resized=None
#            
        if self.i_image_resized!=None:
            self.i_image_resized_reshaped=image_2d_to_3d(self.i_image_resized)
        else:
            self.i_image_resized_reshaped=None
#        
        if self.g_image_resized!=None:
            self.g_image_resized_reshaped=image_2d_to_3d(self.g_image_resized)
        else:
            self.g_image_resized_reshaped=None
#            
    def plot_image(self):
        if self.i_image!=None:
            plt.subplot(121)
            img=plt.imshow(self.i_image)
            img.set_cmap('hot')
            plt.title("I band Image")
            plt.axis('off')
        if self.g_image!=None:
            plt.subplot(122)
            img=plt.imshow(self.g_image)
            plt.title("G band Image")
            img.set_cmap('hot')
            plt.axis('off')
        plt.figure()
        if self.i_image_resized!=None:
            plt.subplot(121)
            img=plt.imshow(self.i_image_resized)
            img.set_cmap('hot')
            plt.title("I band Resized Image")
            plt.axis('off')
        if self.g_image_resized!=None:
            plt.subplot(122)
            img=plt.imshow(self.g_image_resized)
            plt.title("G band Resized Image")
            img.set_cmap('hot')
            plt.axis('off')
   


# In[28]:

def read_target_data_csv(target_data_csv_path):
    #Args: target data csv path
    target_data_df=pd.read_csv(target_data_csv_path,sep=";",dtype={"SDSS_ID":str,'logMstar':np.float64, 'err_logMstar':np.float64, 'Distance':np.float64})
    return target_data_df
    


# In[29]:

def get_Data(target_data_df,image_data_path):
    # Arg: a Dataframe of the target data and the path of the images
    # Returns a list of data objects
#    
    data_objects=[]
    for index in range(len(target_data_df)):
        SDSS_ID=target_data_df.iloc[index]["SDSS_ID"]
        logMstar=target_data_df.iloc[index]["logMstar"]
        err_logMstar=target_data_df.iloc[index]["err_logMstar"]
        Distance=target_data_df.iloc[index]["Distance"]
#
        data_objects.append(data(SDSS_ID,logMstar,err_logMstar,Distance,image_data_path))
    return data_objects

