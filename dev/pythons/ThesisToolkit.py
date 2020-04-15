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
import random

'''
General basic toolkits for building the dataset for CBIR system from Google Street View
    FeaturesToolkit: extracting the features
    FeaturesIOToolkit: Read/Write the features 
'''

class FeaturesIOToolkit:

    def __init__(self):
        pass

    def DelfReadSingleImage(self, features_path = None):
        locations, _, descriptors, _, _ = feature_io.ReadFromFile(features_path)
        name = features_path.split('/')[-1]
        name = name.split('.')[0]
        return (name, locations, descriptors)

    def DelfReadDirectory(self, directory = None, rand = False, num_samples = 0):
        '''
        Read the delf features to the memory
        @InPut: dir - Directory that contains the features in numpy format
        @Output: Numpy array of the features,
                 Array of images name
        '''
        print("Loading DELF features")
        features_file_list = [f for f in os.listdir(directory)]
        if rand:
            if num_samples <= 0:
                print("ERROR: number of samples must > 0")
                sys.error(-1)
            if num_samples > len(features_file_list):
                num_samples = len(features_file_list)
            print("Reading radom %d samples from %d entries" %(num_samples, len(features_file_list)))
            features_file_list = random.sample(features_file_list, k = num_samples)
        descriptors_list = []
        locations_list = []
        image_name_list = [f.split('.')[0] for f in features_file_list]
        N = mp.cpu_count()
        p = mp.Pool(N)
        rets = p.map(DelfReadSingleImage, [os.path.join(directory, f) for f in features_file_list])
        p.close()
        p.terminate()
        for ret in rets:
            image_name_list.append(ret[0])
            locations_list.append(ret[1])
            descriptors_list.append(ret[2])
        # for f in features_file_list:
        #     name, locations, descriptors self.DelfReadSingleImage(os.path.join(directory, f))
        #     descriptors_list.append(descriptors)
        #     locations_list.append(locations)
        #     image_name_list.append(name)
        print("Load %d features" %len(descriptors_list))
        del rets
        return image_name_list, locations_list, descriptors_list

    def SiftReadSingleImage(features_path = None):
        features = np.load(features_path, allow_pickle=True)
        locations = np.float32(features[:,0:2])
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