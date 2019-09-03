import joblib
import sklearn.cluster
import numpy as np
import multiprocessing as mp
import os
import sys
import google_streetview.api
import shutil
import json
import math
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

'''
This is the toolkit for creating the city-scale database with google streetview.
The toolkit includes two class:
Class: DatabaseToolkit
    @Parameters:

    @Atributes:

    @Methods:
        CrawlImages: Crawl the images from desired images
        PostProcessing: Post-process the database
Class: QueryToolKit
    @Parameters:

    @Attributes:

    @Methods:
        AnnotatesQueries:
'''

class DatabaseToolkit:
    const_1m = 0.000008985 # 1 metter different in lat
    const_half_pi = 0.0174533 # pi/180

    def __init__(self, key = None, name = None):
        if(key == None):
            print("WARING: Key is missing, cralwer is inacessible")
        else:
            self.key = key
        if(name == None):
            print("WARING: Name is missing, set default name to StreetView")
            self.name = 'StreetView'
        else:
            self.name = name
    
    def CrawlImages(self, start_lat=None, start_lng=None, end_lat=None, end_lng=None, download_dir = None, density=20):
        if(self.key == None):
            print("ERROR: GoogleMap API requrire a biiling account and a key")
            sys.exit(-1)
        distance = density * const_1m
        lat = start_lat
        index_lat = 0
        index_lng = 0
        indexes = []
        print("Start Crawling The Streetview Images")
        while lat <= end_lat:
            lng = start_lng
            while lng <= end_lng:
                location = str(lat) + ',' + str(lng)
                # define the params for the streetview API
                params = []
                for heading in range(0,360,45):
                    req = {
                        'size': '640x640',
                        'pitch': '15',
                        'key': self.key,
                        'location': location,
                        'heading': str(heading)
                    }
                    params.append(req)
                save_dir = os.path.join(download_dir,str(index_lat) + '_' + str(index_lng))
                # Create the result object
                results = google_streetview.api.results(params)
                # Download images to the ditectory downloads:
                result.download_links(save_dir)
                # Remove the directory with no images and index the downloaded location
                if len(os.listdir(save_dir)) == 1:
                    shutil.rmtree(save_dir)
                else:
                    # Load the metadata from dowloaded images to index the location
                    json_file = open(os.path.join(save_dir, 'metadata.json'))
                    json_data = json.load(json_file)
                    panoid = json_data[0]['panoid']
                    gps = json_data[0]['location']
                    indexes.append((panoid, str(index_lat) + '_' + str(index_lng), lat, lng, gps['lat'], gps['lng'] ))
                lng += distance/math.cos(lat*const_half_pi)
                index_lng += 1
            print("Crawling done with lat = %f" %lat)
            lat += distance
            index_lat += 1


class QueryToolkit:

    const_10m = 0.000109144766254594

    def __init__(self, name=None):
        self.name = name
        pass
    
    def _get_exif_data(image):
        """
        Returns a dictionary from the exif data of an PIL Image item. 
        Also converts the GPS Tags
        """
        exif_data = {}
        info = image.getexif()
        if info:
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    gps_data = {}
                    for t in value:
                        sub_decoded = GPSTAGS.get(t, t)
                        gps_data[sub_decoded] = value[t]

                    exif_data[decoded] = gps_data
                else:
                    exif_data[decoded] = value
        return exif_data

    def _get_if_exist(data, key):
        if key in data:
            return data[key]
            
        return None
        
    def _convert_to_degress(value):
        """Helper function to convert the GPS coordinates stored in the EXIF to degress in float format"""
        d0 = value[0][0]
        d1 = value[0][1]
        d = float(d0) / float(d1)

        m0 = value[1][0]
        m1 = value[1][1]
        m = float(m0) / float(m1)

        s0 = value[2][0]
        s1 = value[2][1]
        s = float(s0) / float(s1)

        return d + (m / 60.0) + (s / 3600.0)

    def _get_lat_lon(exif_data):
        """
        Returns the latitude and longitude, if available, from the provided exif_data 
        (obtained through get_exif_data above)
        """
        lat = None
        lon = None

        if "GPSInfo" in exif_data:		
            gps_info = exif_data["GPSInfo"]

            gps_latitude = _get_if_exist(gps_info, "GPSLatitude")
            gps_latitude_ref = _get_if_exist(gps_info, 'GPSLatitudeRef')
            gps_longitude = _get_if_exist(gps_info, 'GPSLongitude')
            gps_longitude_ref = _get_if_exist(gps_info, 'GPSLongitudeRef')

            if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
                lat = _convert_to_degress(gps_latitude)
                if gps_latitude_ref != "N":                     
                    lat = 0 - lat

                lon = _convert_to_degress(gps_longitude)
                if gps_longitude_ref != "E":
                    lon = 0 - lon

    def AnnotatesQueries(self, query_dir = None, database_indexes = None, query_indexes = None, bound = None):
        '''
        '''
        # Get all GPS location of the query
        for f in os.listdir(query_dir):
            im = Image.open(os.path.join(query_dir,f))
            exif = _get_exif_data(im)
            gps = _get_lat_lon(exif)
            queries.append((f[:-4], gps[0], gps[1]))
        # Get all GPS location of the database using the database index file
        with open(database_indexes, 'r') as fi:
            contents = fi.readlines()
        lines = [line.strip('\n') for line in contents]
        database = []
        for line in lines:
            tup = line.split('\t')
            database.append((tup[0], float(tuo[1]), float(tuo[2])))
        # Assign database images to each query with the distance threshold smaller than bound
        distance_threshold = bound/10.0 *const_10m
        for query in queries:
            rank_list = []
            for entry in database:
                lat_diff = query[1] - entry[1]
                lng_diff = query[2] - entry[2]
                distance = math.sqrt(lat_diff*lat_diff + lng_diff*lng_diff)
                if(distance < distance_threshold):
                    rank_list.append(entry[0])
        # Save to the query index files:
        

