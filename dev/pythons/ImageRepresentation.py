import joblib
import sklearn.cluster
import numpy as np
import multiprocessing as mp
import os
import ThesisToolkit
from ThesisToolkit import FeaturesIOToolkit
from delf import extractor
from sklearn.decomposition import IncrementalPCA

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
featuresIO = FeaturesIOToolkit()
class ImageRepresentation:

    def __init__(self, name=None, saved_model=None, k=None, extractor=None):
        self.name = name
        self.extractor = extractor
        if saved_model == None:
            self.codebook = sklearn.cluster.MiniBatchKMeans(n_clusters=k, random_state = 0)
            self.weighted_vector = None # adding weighted_vector
        else: # create the code book from saved model
            self.codebook = joblib.load(saved_model)

    def CreateCodeBookFromFeatures(self, features_dir=None, max_entries = None, feature_type=None, 
                                   chunk_size = None, training_epocs=None, save_dir=None):
        '''
        Create the visual words by running Kmean on the features
        @Input: The features bag
        @Output: The codebook
        '''
        # Load all features in the directory to the list of features
        if max_entries:
            rand = True
        else:
            rand = False
        if feature_type == 'sift':
            _, _, descriptors_list = featuresIO.SiftReadDirectory(features_dir)
        elif feature_type == 'delf':
            _, _, descriptors_list = featuresIO.DelfReadDirectory(features_dir, rand = rand, num_samples = max_entries)
        buffer = []
        index = 0
        # Train the codebook with kmean 
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
        
        # # Calculate weighted vector with tf-idf
        # database = np.concatenate(descriptors_list, axis = 0)



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
        if not os.path.exists(database_dir):
            os.makedirs(database_dir)
        p.starmap(np.save, [(os.path.join(database_dir, image), vector) for vector, image in zip(bow_vectors, image_list)])
        # for vector, image in zip(bow_vectors, image_list):
        #     output_file = os.path.join(database_dir, image)
        #     np.save(output_file, vector)
    

    def VLADEncode(self, image_features, pca = False, transformer = None):
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
        # Power normalization - SSR
        concat_vector = np.sign(concat_vector)*np.sqrt(np.abs(concat_vector))
        if pca:
            # Load the PCA model
            reduced_vector = transformer.transform([concat_vector])[0]
        else:
            reduced_vector = concat_vector
        # L2 normalize
        l2_norm = np.linalg.norm(reduced_vector)
        vlad_vector = reduced_vector/l2_norm

        # print(vlad_vector.shape)
        # PCA and whitening 

        return vlad_vector

    def MP_VLADEncode(self, image_features, image_locs, grid = 1, pca = False, transformer = None):
        '''
        Encode the image with VLAD 
        '''
        # patches layout
        # grid: 0 --> fixed non-overlapped
        #       1 --> fixed overlapped
        #       2 --> multi-scale
        #       3 --> region proposal
        grid = 1
        if grid == 0:
            patches = [(0,0,320,320),
                    (320,0,640,320),
                    (0,320,320,640),
                    (320,320,640,640)]
        else:
            patches = [(0,0,426,426),
                    (213,0,640,213),
                    (0,213,426,640),
                    (213,213,640,640)]
        # create a set of features for each patches
        features_patches = [[],[],[],[]]
        loc_patches = [[],[],[],[]]
        for loc, des in zip(image_locs, image_features):
            for i in range(len(patches)):
                if(loc[0] >= patches[i][0] and loc[0] < patches[i][2] and loc[1] >= patches[i][1] and loc[1] < patches[i][3]):
                    features_patches[i].append(des)
                    loc_patches[i].append(loc)
        vlad_patches = []
        for features_patch in features_patches:
            vlad_patch = self.VLADEncode(image_features = np.array(features_patch), pca = pca, transformer = transformer)
            vlad_patches.append(vlad_patch)
        mpvlad_vec = np.ravel(vlad_patches)
        # return mpvlad_vec
        # Adding first level of pyramid
        image_vlad = self.VLADEncode(image_features = image_features, pca = pca, transformer = transformer)
        spp_vlad = np.concatenate([image_vlad, mpvlad_vec])
        print(spp_vlad.shape)

        return spp_vlad 
    
    def BuildVLADDatabase(self, 
                          features_dir=None, 
                          feature_type=None, 
                          database_dir=None, 
                          mpvlad = False, 
                          grid = 1, 
                          pca = False, 
                          transformer = None):
        '''
        Create a database with VLAD
        @Input: a folder of image descriptor
        '''
        print("Creating VLAD dataset at %s with feature directory %s" %(database_dir, features_dir))
        if pca:
            print("PCA enable")
            print(transformer.get_params())
        if mpvlad:
            print("Multi-paches vlad enable")
        # Create the thread pool for encoding in parallel
        N = mp.cpu_count()
        # Load all features in the directory to the list of features
        loader = None
        if feature_type == 'sift':
            # image_list, locs_list, descriptors_list = featuresIO.SiftReadDirectory(features_dir)
            loader = featuresIO.SiftReadSingleImage
        elif feature_type == 'delf':
            # image_list, locs_list, descriptors_list = featuresIO.DelfReadDirectory(features_dir)
            loader = featuresIO.DelfReadSingleImage
        chunk_size = 20000
        entries = os.listdir(features_dir)
        feat_full_list = []
        feat_chunk_list = []
        for f in entries:
            feat_path = os.path.join(features_dir, f)
            feat_chunk_list.append(feat_path)
            if len(feat_chunk_list) > chunk_size:
                feat_full_list.append(feat_chunk_list)
                feat_chunk_list = []
        if len(feat_chunk_list) > 0:
            feat_full_list.append(feat_chunk_list)
        
        # Chunks processing
        for i in range(len(feat_full_list)):
            print("Proccessing %d of %d chunks" %(i,len(feat_full_list)))
            p = mp.Pool(processes=N)
            feat_list = feat_full_list[i]
            descriptors_list = []
            locs_list = []
            image_list = []
            rets = p.map(loader, feat_list)
            for ret in rets:
                descriptors_list.append(ret[2])
                locs_list.append(ret[1])
                image_list.append(ret[0])
            del rets
            if mpvlad:
                encoding = self.MP_VLADEncode
                args_list = [(descriptors, locs, grid, pca, transformer) for descriptors, locs in zip(descriptors_list, locs_list)]
            else: 
                encoding = self.VLADEncode
                args_list = [(descriptors, pca, transformer) for descriptors in descriptors_list]
            vlad_vectors = p.starmap(encoding, args_list)
            # Save the features
            if not os.path.exists(database_dir):
                os.makedirs(database_dir)
            p.starmap(np.save, [(os.path.join(database_dir, image), vector) for vector, image in zip(vlad_vectors, image_list)])
            p.terminate()
            p.close()
            del descriptors_list
            del locs_list
        # for vector, image in zip(vlad_vectors, image_list):
        #     output_file = os.path.join(database_dir, image)
        #     np.save(output_file, vector)

    def meanPooling(self, image_features):
        mean_pooling = np.mean(image_features, axis=0)
        dot_prod = mean_pooling.dot(mean_pooling)
        mean_pooling /= np.sqrt(dot_prod)
        return mean_pooling

    def maxPooling(self, image_features):
        max_pooling = np.amax(image_features, axis=0)
        dot_prod = max_pooling.dot(max_pooling)
        max_pooling /= np.sqrt(dot_prod)
        return max_pooling

    def sumPooling(self, image_features):
        sum_pooling = np.sum(image_features, axis=0)
        dot_prod = sum_pooling.dot(sum_pooling)
        sum_pooling /= np.sqrt(dot_prod)
        return sum_pooling

    def GeMPooling(self, image_features):
        p = np.power(np.abs(image_features), 0.5)*np.sign(image_features)
        gem_pooling = np.mean(p, axis=0)
        gem_pooling = np.float_power(np.abs(gem_pooling), 2)*np.sign(gem_pooling)
        dot_prod = gem_pooling.dot(gem_pooling)
        gem_pooling /= np.sqrt(dot_prod)
        return gem_pooling

    def BuildTestDatabase(self, features_dir=None, feature_type=None, database_dir=None):
        '''
        Create a database with VLAD
        @Input: a folder of image descriptor
        '''
        print("Creating Test dataset at %s with feature directory %s" %(database_dir, features_dir))
        # Create the thread pool for encoding in parallel
        N = mp.cpu_count()
        # Load all features in the directory to the list of features
        loader = None
        if feature_type == 'sift':
            # image_list, locs_list, descriptors_list = featuresIO.SiftReadDirectory(features_dir)
            loader = featuresIO.SiftReadSingleImage
        elif feature_type == 'delf':
            # image_list, locs_list, descriptors_list = featuresIO.DelfReadDirectory(features_dir)
            loader = featuresIO.DelfReadSingleImage
        chunk_size = 20000
        entries = os.listdir(features_dir)
        feat_full_list = []
        feat_chunk_list = []
        for f in entries:
            feat_path = os.path.join(features_dir, f)
            feat_chunk_list.append(feat_path)
            if len(feat_chunk_list) > chunk_size:
                feat_full_list.append(feat_chunk_list)
                feat_chunk_list = []
        if len(feat_chunk_list) > 0:
            feat_full_list.append(feat_chunk_list)
        
        # Chunks processing
        database_max = []
        database_mean = []
        database_sum = []
        database_gem = []
        indexes = []
        if not os.path.exists(database_dir):
            os.makedirs(database_dir)
        for i in range(len(feat_full_list)):
            print("Proccessing %d of %d chunks" %(i,len(feat_full_list)))
            p = mp.Pool(processes=N)
            feat_list = feat_full_list[i]
            descriptors_list = []
            locs_list = []
            image_list = []
            rets = p.map(loader, feat_list)
            for ret in rets:
                descriptors_list.append(ret[2])
                indexes.append(ret[0])
            del rets
            # vectors_max = p.map(self.maxPooling, descriptors_list)
            # vectors_mean = p.map(self.meanPooling, descriptors_list)
            vectors_gem = p.map(self.GeMPooling, descriptors_list)
            # for v1,v2 in zip(vectors_max,vectors_mean):
            #     database_max.append(v1)
            #     database_mean.append(v2)
            for v in vectors_gem:
                database_gem.append(v)
            p.terminate()
            p.close()
            del descriptors_list
            del locs_list
            # del vectors_max
            # del vectors_mean
            del vectors_gem
        # np.save(os.path.join(database_dir, "database_max"), database_max)
        # np.save(os.path.join(database_dir, "database_mean"), database_mean)
        np.save(os.path.join(database_dir, "database_gem"), database_gem)
        np.save(os.path.join(database_dir, "indexes_gem"), indexes)

    def FVEncode(self, image_features):
        '''
        Encode the image with Fisher Vector
        '''

