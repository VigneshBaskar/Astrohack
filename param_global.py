# Parameters File
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import cv2
import pickle


astrohack_path=os.path.dirname(os.path.realpath(__file__))

data_path=os.path.join(astrohack_path,"Data")
sample_data_path=os.path.join(data_path,"SAMPLE")

full_data_path="/scratch/leuven/sys/ASTROHACK_DATA/Train"

code_path=os.path.join(astrohack_path,"Code")
neural_network_path=os.path.join(code_path,"Neural_Network")
pre_processing_path=os.path.join(code_path,"Pre_processing")
predictions_path=os.path.join(code_path,"Predicitons")

output_path="/data/leuven/319/vsc31950/output_data"
