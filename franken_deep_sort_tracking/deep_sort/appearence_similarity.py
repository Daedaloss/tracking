#tensorflow imports
from statistics import mode
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import random

from sqlalchemy import true
import tensorflow as tf
from pathlib import Path
from keras import models, Model
from keras.applications import vgg16, MobileNetV3Small, EfficientNetB2
from keras.applications.vgg16 import VGG16, preprocess_input 
from keras.layers import Input, Flatten, Dense
import scipy.spatial.distance as spatialDistance
from annoy import AnnoyIndex
import ssl

#utils imports
import pickle
from distutils.version import LooseVersion
import tqdm
from scipy.ndimage import center_of_mass
from skimage.transform import rescale
import skimage
import skvideo
import skvideo.io
from joblib import Parallel, delayed
import cv2
from colormap import hex2rgb
import copy
import sys
import time
from shapely.geometry import Polygon
skvideo.setFFmpegPath("/usr/local/bin/")
import skvideo.io
from scipy.optimize import linear_sum_assignment
from skimage.io import imread
from skimage.feature import hog
import matplotlib.pyplot as plt

from utils import * 

target_shape = (200, 200)



class Appearence_Similarity:
    """
    This class handles feature detection and comparison.

    Parameters
    ----------


    Attributes
    ----------


    """

    def __init__(self, model_type=None):
        self.model_type = model_type

        if model_type=='vgg':
            self.model = vgg16.VGG16(include_top=True, weights='imagenet')
            self.model.summary()
        elif model_type == 'mobileNet':
            self.model = MobileNetV3Small(weights='imagenet', minimalistic=True)
            self.model.summary()
        elif model_type == 'efficientNet':
            self.model = EfficientNetB2(weights='imagenet')
            self.model.summary()
            
        else:
            self.model = None 
    
    def get_HOG(self, rescaled_img):
        #hog extraction 
        #rescaled_img = rescale_img(bbox, frame)
        print(rescaled_img.shape)
        try:
            out = hog(rescaled_img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=False, channel_axis=-1, feature_vector=True)
            #print(out)
        except Exception as e:
            print(e) 
            out = []
        return out 

    def get_cnn_features(self, rescaled_img):

        try:
            x = np.expand_dims(rescaled_img,axis=0)
            x = preprocess_input(rescaled_img)
            x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
            features= self.model.predict(x)
            return features[0, :]
        except Exception as e :
            print(e)
            return []
    

    def get_features(self, frame, bbox):

       

        if self.model_type == "HOG":
            rescaled_img = rescale_img(bbox, frame)
            features = self.get_HOG(rescaled_img)
        elif self.model_type == "vgg":
            rescaled_img = rescale_img(bbox, frame, mask_size=224)
            features = self.get_cnn_features(rescaled_img)
        elif self.model_type == "mobileNet":
            rescaled_img = rescale_img(bbox, frame, mask_size=224)
            features = self.get_cnn_features(rescaled_img)
        elif self.model_type == "efficientNet":
            rescaled_img = rescale_img(bbox, frame, mask_size=260)
            features = self.get_cnn_features(rescaled_img)
        else:
            features = []
        
        return features 
