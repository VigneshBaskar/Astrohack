# Parameters File
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import cv2
import pickle
import keras
import random

image_size = 32

astrohack_path=os.path.dirname(os.path.realpath(__file__))

data_path=os.path.join(astrohack_path,"Data")

# legacy
sample_data_path=os.path.join(data_path,"SAMPLE")

# original input path
full_data_path="/scratch/leuven/sys/ASTROHACK_DATA/Train"
test_data_path="/scratch/leuven/sys/ASTROHACK_DATA/Test"

code_path=os.path.join(astrohack_path,"Code")
neural_network_path=os.path.join(code_path,"Neural_Network")
pre_processing_path=os.path.join(code_path,"Pre_processing")
predictions_path=os.path.join(code_path,"Predicitons")

# model results
output_path="/data/leuven/319/vsc31950/output_data"

# transformed dataset
temp_path="/scratch/leuven/319/vsc31950/output_files"
