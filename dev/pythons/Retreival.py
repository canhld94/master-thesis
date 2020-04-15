'''
Retrieval benchmark
'''

import ImageRepresentation
import ThesisToolkit.FeatutesIOToolkit as featuresIO
import sys

class Retrieval:
    def __init__(self, database_vectors=None, image_representation = None, database_images = None, database_features = None):
        if database_vectors == None:
            print("ERROR: database vectors is missing")
            sys.exit(-1)
        else: 
            self.database_vectors = database_vectors
        if image_representation = None:
            print("ERROR: image represenation object is missing")
            sys.exit(-1)
        else:
            self.ir = image_representation
        if database_images = None:
            print("WARNING: database images is missing, visual retrieval is not available")
        if database_features == None:
            print("WARNING: database fetures is missng, geometric verification is not available")
        self.database_features = database_features
        self.database_images = database_images
    
    def runSingleImage(self, image_path = None, geometric_verification = 0, visual_result = 0):
        pass

    def runBenchmark(self, query_dir = None, geometric_verification = 0):
        pass