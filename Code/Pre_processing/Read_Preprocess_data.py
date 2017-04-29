
# coding: utf-8

# In[22]:

#get_ipython().magic('matplotlib inline')

from Data_Preparation_Library import *
from multiprocessing.dummy import Pool as ThreadPool

# In[30]:

# Read the target data csv which has SDSS_ID and distance along with logMstar

#target_data_csv_path=os.path.join(data_path,"sample.csv")
target_data_csv_path=os.path.join(full_data_path,"..","Train.csv")
input_target_data_df=read_target_data_csv(target_data_csv_path)

# create partitioning columsn
numPartitions = 76
partitionArray = np.array(range(len(input_target_data_df))) % numPartitions
np.random.shuffle(partitionArray)
input_target_data_df["partition"] = partitionArray

def extractBatch(i):
  batch = input_target_data_df[input_target_data_df["partition"]==i]
  n = len(batch)
  print("Extracting batch " + str(i) + ". Number of images: " +  str(n))
  input_data_object=get_Data(batch,full_data_path)
#with open(os.path.join(input_data_path,'input_data_object.p'), 'wb') as handle:
  with open(os.path.join(temp_path,'full_data_object_' + str(i) + '.p'), 'wb') as handle:
    pickle.dump(input_data_object, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
# Open the urls in their own threads
# and return the results
pool = ThreadPool(20) 
#close the pool and wait for the work to finish 
# Pass on this dataframe of the targetdata and the image data directory and get back a list of objects of the data
pool.map(extractBatch, np.array(range(numPartitions)))

pool.close() 
pool.join() 

# In[32]:

# print(input_data_object[0].i_image_resized.reshape(1,-1).shape)
# print(input_data_object[0].g_image_resized.reshape(1,-1).shape)


# In[46]:

#input_target_data_df

