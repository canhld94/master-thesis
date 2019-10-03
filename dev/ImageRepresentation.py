import joblib
import sklearn.cluster
import numpy as np
import multiprocessing as mp
import os
import ThesisToolkit
from ThesisToolkit import FeaturesIOToolkit as featuresIO
from delf import extractor
'''
Image representation class, which provide functions to represent the image 
@Parameter:
    name: name of the object
    saved_model: file name of the save KMean model
    k: number of words to construct the codeword, only applicable if saved_model == None
@Atrribute:
    name: name of the object
    extractor: type of the local features
    codebook: Sklearn Kmean object, use as the main codebook
@Methods:
    CreateCodeBookFromFeatures
    SaveModel
    BoWEncode
    VLADEncode
    FVEncode
'''

class ImageRepresentation:

    def __init__(self, name=None, saved_model=None, k=None, extractor=None):
        self.name = name
        self.extractor = extractor
        if saved_model == None:
            self.codebook = sklearn.cluster.MiniBatchKMeans(n_clusters=k, random_state = 0)
        else: # create the code book from saved model
            self.codebook = joblib.load(saved_model)

    def CreateCodeBookFromFeatures(self, features_dir=None, feature_type=None, chunk_size = None, training_epocs=None, save_dir=None):
        '''
        Create the visual words by running Kmean on the features
        @Input: The features bag
        @Output: The codebook
        '''
        # Load all features in the directory to the list of features
        if feature_type == 'sift':
            _, _, descriptors_list = featuresIO.ReadRootSiftFeatures(features_dir)
        elif feature_type == 'delf':
            _, _, descriptors_list = featuresIO.ReadDelfFeatures(features_dir)
        buffer = []
        index = 0
        for epoc in range(training_epocs):
            print("Training codebooks with epoc %d" % epoc)
            for des in descriptors_list:
                index += 1
                buffer.append(des)
                if index % chunk_size == 0:
                    data = np.concatenate(buffer, axis = 0)
                    self.codebook.partial_fit(data)
                    print("Partial fit of %d out of %d, inertia = %f" %(index, len(descriptors_list), self.codebook.inertia_))
                    buffer = []
            if save_dir:
                joblib.dump(self.codebook, os.path.join(save_dir, self.name + '_model_epoc_' + str(epoc) + '.pkl'))


    def SaveModel(self, output_dir):
        '''
        Save the kmean model to the output directory
        '''
        joblib.dump(self.codebook, os.path.join(save_dir, self.name + '_model.pkl'))

    
    def BoWEncode(self, image_features):
        '''
        Encode the image features by histogram of word and L2 norm 
        '''
        indexes = self.codebook.predict(image_features)
        hist, _ = np.histogram(indexes, bins = range(self.codebook.cluster_centers_.shape[0] + 1))
        # print(hist.shape)
        l2_norm = np.linalg.norm(hist)
        bow_vector = hist/l2_norm
        print(bow_vector.sum())
        return bow_vector

    def BuildBoWDatabase(self, features_dir=None, feature_type=None, database_dir=None):
        '''
        Create a database with BoW 
        @Input: a folder of image descriptor
        '''
        print("Creating BoW dataset at %s with feature directory %s" %(database_dir, features_dir))
        # Load all features in the directory to the list of features
        if feature_type == 'sift':
            image_list, _, descriptors_list = featuresIO.SiftReadDirectory(features_dir)
        elif feature_type == 'delf':
            image_list, _, descriptors_list = featuresIO.DelfReadDirectory(features_dir)
        # Create the thread pool for encoding in parallel
        N = mp.cpu_count()
        p = mp.Pool(processes=N)
        bow_vectors = p.map(self.BoWEncode, [descriptors for descriptors in descriptors_list])
        # Save the features
        for vector, image in zip(bow_vectors, image_list):
            output_file = os.path.join(database_dir, image)
            np.save(output_file, vector)
    

    def VLADEncode(self, image_features):
        '''
        Encode the image with VLAD 
        '''
        indexes = self.codebook.predict(image_features)
        # Create an empty kxD vlad features
        vlad = [np.array([0.0]*self.codebook.cluster_centers_.shape[1]) for _ in range(self.codebook.cluster_centers_.shape[0])]
        # Calculate VLAD vector by adding residual of features to centroid in each centroid
        for index, feature in zip(indexes, image_features):
            vlad[index] += feature - self.codebook.cluster_centers_[index]
        # TODO: Intra-normalization
        for vector in vlad:
            vector = np.sign(vector)*np.sqrt(np.abs(vector))
            intra_l2_norm = np.linalg.norm(vector)
            if(intra_l2_norm > 0.0):
                vector = vector/intra_l2_norm
        # Concatenate into 1-D array
        concat_vector = np.ravel(vlad)
        # Power normalization
        concat_vector = np.sign(concat_vector)*np.sqrt(np.abs(concat_vector))
        # L2 normalize
        l2_norm = np.linalg.norm(concat_vector)
        vlad_vector = concat_vector/l2_norm
        print(vlad_vector.shape)
        # PCA and whitening 

        return vlad_vector
    
    def BuildVLADDatabase(self, features_dir=None, feature_type=None, database_dir=None):
        '''
        Create a database with VLAD
        @Input: a folder of image descriptor
        '''
        print("Creating VLAD dataset at %s with feature directory %s" %(database_dir, features_dir))
        # Load all features in the directory to the list of features
        if feature_type == 'sift':
            image_list, _, descriptors_list = featuresIO.SiftReadDirectory(features_dir)
        elif feature_type == 'delf':
            image_list, _, descriptors_list = featuresIO.DelfReadDirectory(features_dir)
        # Create the thread pool for encoding in parallel
        N = mp.cpu_count()
        p = mp.Pool(processes=N)
        vlad_vectors = p.map(self.VLADEncode, [descriptors for descriptors in descriptors_list])
        # Save the features
        for vector, image in zip(vlad_vectors, image_list):
            output_file = os.path.join(database_dir, image)
            np.save(output_file, vector)


    def FVEncode(self, image_features):
        '''
        Encode the image with Fisher Vector
        '''

