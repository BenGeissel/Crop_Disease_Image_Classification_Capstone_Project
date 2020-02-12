# Import necessary packages
import pandas as pd
import numpy as np
from PIL.Image import core as image
from PIL import Image



def image_to_array(image_loc):
    '''
    Function to convert every image into (256*256*3) array
    
    Input: Image file path
    
    Output: 256*256*3 array representing the image pixel matrix
    '''
    
    img = Image.open(image_loc)
    arr = np.array(img)
    
    return arr



def pixel_normalization(img_array):
    '''
    Function to normalize pixels (0 to 1) of each image (0 to 255 pixel values possible)
    
    Input: Image matrix with integer values between 0 and 255
    
    Output: Image matrix with float values between 0 and 1
    '''
    
    img_array = img_array.astype('float32')
    img_array /= 255.0
    
    return img_array



def pixel_centering(norm_img_arr):
    '''
    Function to center pixel values based on mean pixel value
    
    Input: Normalized image matrix with float values between 0 and 1
    
    Output: Centered image matrix with float values between 0 and 1 centered around image matrix mean value
    '''
    
    mean = norm_img_arr.mean()
    norm_img_arr = norm_img_arr - mean
    
    return norm_img_arr

# For ML Models
def image_to_flat_array(image_loc):
    '''
    Function to convert every image into (196608,) array
    
    Input: Image file path
    
    Output: (196608,) array representing the image pixel matrix
    '''
    
    img = Image.open(image_loc)
    arr = np.array(img)
    flat_arr = arr.flatten()
    
    return flat_arr


def image_prediction(img_path, model, class_map_dict):
    '''
    Function to predict image class
    
    Input: Image file path, CNN Keras Model
    
    Output: Image Class Prediction
    '''
    
    arr = image_to_array(img_path)
    norm = pixel_normalization(arr)
    final = pixel_centering(norm).reshape((1, 256, 256, 3))
    pred = model.predict(final)
    result_dict = dict(enumerate(pred[0]))
    int_class = max(result_dict, key = result_dict.get)
    crop_leaf_class = class_map_dict[int_class]
    
    return crop_leaf_class