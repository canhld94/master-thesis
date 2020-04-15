# Copyright 2017 Nitish Mutha (nitishmutha.com)
# Copyright 2019 Duc Canh Le

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import pi
import numpy as np
import PIL
from PIL import Image
import multiprocessing as mp
import os

def openIm(img_path):
    img_name = img_path.split('/')[-1][:-11]
    return Image.open(img_path), img_name


class PanoTool():
    def __init__(self, height=640, width=640, frame_width=6656, frame_height=3328, frame_channel=3):
        self.FOV = [0.25, 0.5]
        self.PI = pi
        self.PI_2 = pi * 0.5
        self.PI2 = pi * 2.0
        self.height = height
        self.width = width
        self.screen_points = self._get_screen_img()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_channel = frame_channel

    def _get_coord_rad(self, isCenterPt, center_point=None):
        return (center_point * 2 - 1) * np.array([self.PI, self.PI_2]) \
            if isCenterPt \
            else \
            (self.screen_points * 2 - 1) * np.array([self.PI, self.PI_2]) * (
                np.ones(self.screen_points.shape) * self.FOV)

    def _get_screen_img(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width), np.linspace(0, 1, self.height))
        return np.array([xx.ravel(), yy.ravel()]).T

    def _calcSphericaltoGnomonic(self, convertedScreenCoord, cp):
        x = convertedScreenCoord.T[0]
        y = convertedScreenCoord.T[1]

        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(cos_c * np.sin(cp[1]) + (y * sin_c * np.cos(cp[1])) / rou)
        lon = cp[0] + np.arctan2(x * sin_c, rou * np.cos(cp[1]) * cos_c - y * np.sin(cp[1]) * sin_c)

        lat = (lat / self.PI_2 + 1.) * 0.5
        lon = (lon / self.PI + 1.) * 0.5

        return np.array([lon, lat]).T

    def _bilinear_interpolation(self, frame, screen_coord):
        uf = np.mod(screen_coord.T[0],1) * self.frame_width  # long - width
        vf = np.mod(screen_coord.T[1],1) * self.frame_height  # lat - height

        x0 = np.floor(uf).astype(int)  # coord of pixel to bottom left
        y0 = np.floor(vf).astype(int)
        x2 = np.add(x0, np.ones(uf.shape).astype(int))  # coords of pixel to top right
        y2 = np.add(y0, np.ones(vf.shape).astype(int))

        base_y0 = np.multiply(y0, self.frame_width)
        base_y2 = np.multiply(y2, self.frame_width)

        A_idx = np.add(base_y0, x0)
        B_idx = np.add(base_y2, x0)
        C_idx = np.add(base_y0, x2)
        D_idx = np.add(base_y2, x2)

        flat_img = np.reshape(frame, [-1, self.frame_channel])

        A = np.take(flat_img, A_idx, axis=0)
        B = np.take(flat_img, B_idx, axis=0)
        C = np.take(flat_img, C_idx, axis=0)
        D = np.take(flat_img, D_idx, axis=0)

        wa = np.multiply(x2 - uf, y2 - vf)
        wb = np.multiply(x2 - uf, vf - y0)
        wc = np.multiply(uf - x0, y2 - vf)
        wd = np.multiply(uf - x0, vf - y0)

        # interpolate
        AA = np.multiply(A, np.array([wa, wa, wa]).T)
        BB = np.multiply(B, np.array([wb, wb, wb]).T)
        CC = np.multiply(C, np.array([wc, wc, wc]).T)
        DD = np.multiply(D, np.array([wd, wd, wd]).T)
        nfov = np.reshape(np.round(AA + BB + CC + DD).astype(np.uint8), [self.height, self.width, 3])
        img = Image.fromarray(nfov)
        # img.save("test.jpg")
        return img

    def toNFOV(self, imgs):
        frame = imgs[0]
        name = imgs[1]
        pitch = 0.4
        yaws = [i*0.125 for i in range(8)]
        imgs = []
        for yaw in yaws:
            center_point = np.array([yaw, pitch])
            cp = self._get_coord_rad(center_point=center_point, isCenterPt=True)
            convertedScreenCoord = self._get_coord_rad(isCenterPt=False)
            spericalCoord = self._calcSphericaltoGnomonic(convertedScreenCoord, cp)
            imgs.append(self._bilinear_interpolation(frame ,spericalCoord))
        return imgs, name            
    
def saveImg(p_img, database_dir):
    imgs = p_img[0]
    panoid = p_img[1]
    for i in range(len(imgs)):
        img_name = '_' + panoid + '_' + str(i) + '.jpg'
        img_path = os.path.join(database_dir, img_name)
        imgs[i].save(img_path)


def BuildingDatabase(panorama_dir=None, database_dir=None):
    """
    Create database of images from the panorama database
    """
    import time

    if not os.path.exists(database_dir):
        os.makedirs(database_dir)
    entries = os.listdir(panorama_dir)
    img_full_list = []
    img_chunk_list = []
    # create list of images
    # As the number of image is quite large (>30000), we should split the
    # list of images to chunks, here I choose 500
    for f in entries:
        img_dir = os.path.join(panorama_dir,f)
        img_name = f[1:] + '_zoom_4.jpg'
        img_path = os.path.join(img_dir, img_name)
        if(os.path.exists(img_path)):
            img_chunk_list.append(img_path)
        if(len(img_chunk_list) >= 200):
            img_full_list.append(img_chunk_list)
            img_chunk_list = []
    if len(img_chunk_list) > 0:
        img_full_list.append(img_chunk_list)
    
    print('Read %d chunk of images' %len(img_full_list))
    # ok now process each chunks of images
    for img_list in img_full_list:
        # read the images, here paralize its:
        start = time.time()
        N = mp.cpu_count()
        p = mp.Pool(processes=N)
        print(len(img_list))
        imgs = p.map(openIm, img_list)
        # print(imgs)

        # Process the images
        pn = PanoTool()
        p_imgs = p.map(pn.toNFOV, imgs)
         
        # Save the imgages
        p.starmap(saveImg, [(p_img, database_dir) for p_img in p_imgs])
        p.close()
        p.terminate()
        end = time.time()
        print("process %d images in %d seconds" %(len(img_list), end - start))
        del imgs
        del p_imgs
        del pn
    pass

# # test the class
if __name__ == '__main__':
    import imageio as im
    img = im.imread('test/pano_test.jpg')
    nfov = PanoTool()
    # Center point: x --> yaw --> 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875,1
    #               y --> pitch --> lower is higher --> 0.4
    center_point = np.array([0.5, 0.4])  # camera center point (valid range [0,1])
    nfov.toNFOV(img, center_point)