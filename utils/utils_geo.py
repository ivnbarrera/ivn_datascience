from time import sleep
import datetime as dt
import pandas as pd
import numpy as np
import math
import itertools
import re
from itertools import compress
from nltk.corpus import wordnet
import nltk
import matplotlib.pyplot as plt


def haversine_distance(latitude_job, longitude_job, latitude_cand, longitude_cand):
    """
    haversine_distance
    ints
    
    returns distance in miles
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [longitude_job, latitude_job, longitude_cand, latitude_cand])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    # earth radius in kilometers
    earth_radius = 6367
    #calculates distances in miles between two points
    
    # calculate distance through haversine function
    distance = 0.621371 * (earth_radius * c)

    # return distance
    return distance

def do_geocode(address, geolocator, attempts=0):
    """ Function to get geolocation from address, recursive
    """
    try:
        return geolocator.geocode(address)
    except:
        if attempts < 2:  # max recursions
            sleep(2)
            return do_geocode(address, geolocator, attempts=attempts+1)

def search_by_address(row, total_address, geolocator):
    """ Function to get geolocation from address in DataFrame
    """
    if total_address in row.keys():
        address = row[total_address]
        sleep(1)
        geo = do_geocode(address, geolocator)
    else:
        print('No Address row')
        pass
     
    if 'latitude' in row.keys() and 'longitude' in row.keys():
        if geo:
            row.longitude = geo.longitude
            row.latitude = geo.latitude
    else:
        print('No Row latitude or longitude')
    return row
    
def search_by_zipcode(row, search):
    """ Function to get geolocation from zip code in DataFrame
    """
    if 'Zip_Code' in row.keys():
        geo = search.by_zipcode(row['Zip_Code'])
        if 'latitude' in row.keys() and 'longitude' in row.keys():
            row.longitude = geo.lng
            row.latitude = geo.lat
        else:
            print('No Row latitude or longitude')
    else: 
        print("No Row Zip_Code")
    
    sleep(1)
    return row