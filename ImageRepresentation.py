'''
Bag of visual world classes
'''
import joblib
import sklearn.cluster
import numpy as np
import multiprocessing as mp
import os
'''
Image representation class, which provide functions to represent the image 
@Parameter:
name: name of the object
saved_model: file name of the save KMean model
k: number of words to construct the codeword, only applicable if saved_model == None
@Atrribute:
name: name of the object
codebook: Sklearn Kmean object, use as the main codebook
@Methods:
CreateCodeBookFromFeatures
SaveModel
BoWEncode
VLADEncode
FVEncode

'''

class ImageRepresentation:

    def __init__(self, name=None, saved_model=None, k=None):
        self.name = name
        if saved_model == None:
            self.codebook = sklearn.cluster.MiniBatchKMeans(n_clusters=k, random_state = 0)
        else: # create the code book from saved model
            self.codebook = joblib.load(saved_model)


    def CreateCodeBookFromFeatures(self, features=None, chunk_size = None, traning_epocs=None, save_dir=None):
        '''
        Create the visual words by running Kmean on the features
        @Input: The features bag
        @Output: The codebook
        '''
        buffer = []
        index = 0
        for epoc in training_epocs:
            print("Training codebooks with epoc %d" % epoc)
            for des in feature:
                index += 1
                buffer.append(des)
                if index % chunk_size == 0:
                    data = np.concatenate(buffer, axis = 0)
                    self.codebook.partial_fit(data)
                    print("Partial fit of %d out of %d, inertia = %f" %(index, len(features_bag), kmeans.inertia_))
                    buffer = []
            if save_model:
                joblib.dump(self.codebook, os.path.join(save_dir, self.name + '_model_epoc_' + str(epoc) + '.pkl'))


    def SaveModel(self, output_dir):
        '''
        Save the kmean model to the output directory
        '''
        joblib.dump(self.codebook, os.path.join(save_dir, self.name + '_model.pkl'))

    
    def BoWEncode(self, image_features):
        '''
        Encode the image features by histogram of world and L2 norm 
        '''
        indexes = self.codebook.predict(image_features)
        hist, _ = np.histogram(indexes, bins = range(self.codebook.cluster_centers_.shape[0] + 1))
        # print(hist.shape)
        l2_norm = np.linalg.norm(hist)
        bow_vector = hist/l2_norm
        # print(bow_vector.shape)
        return bow_vector

    def __ReadRootSiftFeatures(self, dir):
        '''
        Read the rSIFT feature to the memory
        @Input: dir - Diretctory that contains the features in npy format 
        @Output: Npy array of features
                 Array of images name
        '''
        print("Loading features")
        features_file_list = [f for f in os.listdir(dir)]
        features_list = []
        image_name_list = []
        for f in features_file_list:
            image_features = np.load(os.path.join(dir, f), allow_pickle=True)
            image_descriptors = np.float32(image_features[:, 2:])
            features_list.append(image_descriptors)
            image_name_list.append(f.split('.')[0])
        print("Loaded  %d features" %len(features_list))
        return image_name_list, features_list

            
    def BuildBoWDatabase(self, features_dir, database_dir):
        '''
        Create a database with BoW 
        @Input: a folder of image descriptor
        '''
        print("Creating BoW dataset at %s with feature directory %s" %(database_dir, features_dir))
        # Load all features in the directory to the list of features
        image_list, features_list = self.__ReadRootSiftFeatures(features_dir)
        # Create the thread pool for encoding in parallel
        N = mp.cpu_count()
        p = mp.Pool(processes=N)
        bow_vectors = p.map(self.BoWEncode, [features for features in features_list])
        # Save the features
        for vector, image in zip(bow_vectors, image_list):
            output_file = os.path.join(database_dir, image)
            np.save(output_file, f)
    


    def VLADEncode(self, image_features):
        '''
        Encode the image with VLAD 
        '''
        indexes = self.codebook.predict(image_features)
        # Create an empty kxD vlad features
        vlad_vector = [np.array([0]*self.codebook.cluster_centers_.shape[1]) for _ in range(codebook.cluster_centers_.shape[0])]
        # Calculate VLAD vector by adding residual of features to centroid in each centroid
        for index, feature in zip(indexes, image_features):
            vlad_vector[index] += feature - self.codebook.cluster_centers[index]
        
        # PCA and whitening 

        return vlad_vector
    

    def FVEncode(self, image_features):
        '''
        Encode the image with Fisher Vector
        '''

