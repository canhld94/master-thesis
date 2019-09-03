'''
Read the rootSIFT descriptor from numpy files and cluster it

'''
import argparse
import sys
import os
import time
import cv2 as cv
import numpy as np
import sklearn.cluster
import sklearn.externals
import joblib
from tensorflow.python.platform import app


cmd_args = None

def _ReadRootSiftFeature(dir):
    '''
    Read the rSIFT feature to the memory
    @Input: dir - Diretctory that contains the features in npy format 
    @Output: Npy array of features
    '''
    print("Loading features")
    features_file_list = [f for f in os.listdir(dir)]
    features = []
    for f in features_file_list:
        image_features = np.load(os.path.join(dir, f), allow_pickle=True)
        image_descriptors = np.float32(image_features[:, 2:])
        features.append(image_descriptors)
    print("Loaded  %d features" %len(features))
    return features


def _ClusteringAndSave(k, features_bag, dir):
    '''
    Perform kmean clustering on the features bag
    @Input: k - number of visaul words
            features_bag: npy array of the features
    @Output: a vocabulary of visual word with ID and centrois 
    '''
    print("Learning the codebook with %d words" % k)
    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=k, random_state = 0, verbose = True)
    buffer = []
    index = 0
    # Loop over the images, for each 50 images then we compute the kmean
    print("Training the codebook")
    for epoc in range(6):
        print("Epoc %d" % epoc)
        for feature in features_bag:
            buffer.append(feature)
            index += 1
            if index % 500 == 0: # We have enough 100 images
                data = np.concatenate(buffer, axis = 0)
                kmeans.partial_fit(data)
                print("Partial fit of %d out of %d, inertia = %f" %(index, len(features_bag), kmeans.inertia_))
                buffer = []
        joblib.dump(kmeans, os.path.join(dir, str(k) + '_model_epoc' + str(epoc) + '.pkl'))


def main(argv):
    features = _ReadRootSiftFeature(cmd_args.features_dir)
    _ClusteringAndSave(cmd_args.k, features, cmd_args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument(
    '--k',
    type=int,
    help="""
    Number of code words
    """)
    parser.add_argument(
    '--features_dir',
    type=str,
    help="""
    Path to the feature directory
    """)
    parser.add_argument(
    '--output_dir',
    type=str,
    help="""
    Path directory to save the model
    """)
    cmd_args, unparsed = parser.parse_known_args()
    app.run(main=main, argv = [sys.argv[0]] + unparsed)
