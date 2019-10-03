import joblib
import sklearn.cluster
import numpy as np
import multiprocessing as mp
import os
import sys
import shutil
import json
import math
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from delf import feature_io

'''
General basic toolkits for building the dataset for CBIR system from Google Street View
    FeaturesToolkit: extracting the features
    FeaturesIOToolkit: Read/Write the features 
'''

class FeaturesIOToolkit:
    def __init__():
        pass

    def DelfReadSingleImage(features_path = None):
        locations, _, descriptors, _, _ = feature_io.ReadFromFile(features_path)
        return locations, descriptors

    def DelfReadDirectory(directory = None):
        '''
        Read the delf features to the memory
        @InPut: dir - Directory that contains the features in numpy format
        @Output: Numpy array of the features,
                 Array of images name
        '''
        print("Loading DELF features")
        features_file_list = [f for f in os.listdir(directory)]
        descriptors_list = []
        locations_list = []
        image_name_list = []
        for f in features_file_list:
            locations, _, descriptors, _, _ = feature_io.ReadFromFile(os.path.join(directory, f))
            descriptors_list.append(descriptors)
            locations_list.append(locations)
            image_name_list.append(f.split('.')[0])
        print("Load %d features" %len(descriptors_list))
        
        return image_name_list, locations_list, descriptors_list

    def SiftReadSingleImage(features_path = None):
        features = np.load(os.path.join(directory, f), allow_pickle=True)
        locations = np.float32(features[:,0:1])
        descriptors = np.float32(features[:, 2:])
        
        return locations, descriptors


    def SiftReadDirectory(directory = None):
        '''
        Read the rSIFT feature to the memory
        @Input: dir - Diretctory that contains the features in npy format 
        @Output: Npy array of features
                 Array of images name
        '''
        print("Loading RSIFT features")
        features_file_list = [f for f in os.listdir(directory)]
        descriptors_list = []
        locations_list = []
        image_name_list = []
        for f in features_file_list:
            features = np.load(os.path.join(directory, f), allow_pickle=True)
            descriptors = np.float32(features[:, 2:])
            locations = np.float32(features[:, 0:1])
            descriptors_list.append(descriptors)
            locations_list.append(locations)
            image_name_list.append(f.split('.')[0])
        print("Loaded  %d features" %len(descriptors_list))
        return image_name_list, locations_list, descriptors_list