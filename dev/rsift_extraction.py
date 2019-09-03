'''
RootSIFT feature extraction with OpenCV
I just thing it was a shame that things have turned out that way

'''

import cv2 as cv
import numpy as np
import os
import argparse
import sys
import multiprocessing as mp
import time

from tensorflow.python.platform import app

cmd_args = None

def _GetImageList(image_paths_file):
    '''
        Get the image paths from a file and save it to the images list
        @Input: file that contain the image paths
        @Output: a list of openCV matrix objects that store the images 
                 a list of file name
    '''
    image_list = []
    image_name_list = []
    # Get all image paths
    with open(image_paths_file, 'r') as input:
        image_paths = input.read().splitlines()
    print(image_paths)

    # Open and read the image, convert it to gray and then add to image list
    for image_path in image_paths:
        im = cv.imread(image_path, flags=cv.IMREAD_UNCHANGED)
        gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
        image_list.append(gray)
        file_name = os.path.basename(image_path)
        image_name = os.path.splitext(file_name)[0]
        image_name_list.append(image_name)

    return image_list, image_name_list


def _GetSingleImageFeature(image):
    '''
        Get the rootSIFT features for a single images
        @Input: an opencv grayscale image object
        @Output: rootSIFT features of the image
    '''
    esp = 1e-7
    # Init the sift detector
    sift = cv.xfeatures2d.SIFT_create(nOctaveLayers = 5, contrastThreshold = 0.05, edgeThreshold = 3,nfeatures = 1000)
    # Find the keypoint and the descriptor with SIFT
    kps, des = sift.detectAndCompute(image, None)
    if not np.shape(kps)[0]:
	    return None, None
    locs = []
    for point in kps:
        temp = (point.pt[1], point.pt[0])
        locs.append(temp)
    print(image.shape, np.shape(kps), np.shape(des))
    # Apply the Hellinger kernel by first L1 normalizing and taking the square-root
    des /= (des.sum(axis=1, keepdims=True) + esp)
    des = np.sqrt(des)
    return (locs, des)

def _GetImageFeatures(images_list):
    '''
        Get the image list and extract rSIFT features from these images
        @Input: a list of opencv mat objects
        @Output: a list of opencv descriptor objects 
    '''
    # Create thread pool to run the extractor on each process
    N = mp.cpu_count()
    p = mp.Pool(processes=N)
    kps_r = []
    des_r = []
    for f in p.map(_GetSingleImageFeature, [image for image in images_list]):
        kps_r.append(f[0])
        des_r.append(f[1])
    p.close()
    return kps_r, des_r


def _SaveImageFeatures(keypoint_list, descriptor_list, image_name_list, output_dir):
    '''
        Get the keypoint and descriptor list and save it to file on each images
        OpenCV keypoint attribute:
            float angle
            int class_id
            int octave
            point2f pt
            float response 
            float size
        @Input: a list of images keypoint, features and a list of corresponding image names
        File format: text
        Feature size
        Keypoint
        Descriptors
    '''
    for keypoints, descriptors, image_name in zip(keypoint_list, descriptor_list, image_name_list):
        if not keypoints:
            continue
        output_file = output_dir + '/' + image_name + '.sift'
        f = np.hstack((keypoints, descriptors))
        np.save(output_file, f)
        


def main(unused_args):
    start = time.time()
    image_list, image_name_list = _GetImageList(cmd_args.image_paths_file)
    print('Load images done in %d sec' %(time.time() - start))
    start = time.time()
    keypoint_list, descriptor_list = _GetImageFeatures(image_list)
    print('Feature extraction done in %d sec' %(time.time() - start))
    start = time.time()
    _SaveImageFeatures(keypoint_list, descriptor_list, image_name_list, cmd_args.output_dir)
    print('Feature saving done in %d sec' %(time.time() - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument(
        '--image_paths_file',
        type = str,
        help = """
        Directory that store images
        """
    )
    parser.add_argument(
        '--output_dir',
        type = str,
        help = """
        Output directory to store image features 
        """
    )
    cmd_args, unparesed = parser.parse_known_args()
    app.run(main = main, argv = [sys.argv[0]] + unparesed)