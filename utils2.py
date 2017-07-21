import tensorflow as tf 
import os
from keras.preprocessing.image import img_to_array, array_to_img, load_img
import numpy as np

#Hyperparameters
img_size = 128
channels = 3
input_dim = img_size*img_size*channels

#Helper function to get all image paths in a given directory   
def get_image_paths(img_dir):
    file=[]
    for subdir, dirs, files in os.walk(img_dir):
        for i in files: 
            i = os.path.join(img_dir, i)
            file.append(i)
        return file

#Helper function to convert image path to numpy array representation of the image
def path_to_numpy(img_dir):
    image = load_img(img_dir, target_size=(img_size, img_size))
    image = img_to_array(image)
    return image
#Function that returns the next batch given the list of all images
def next_batch(dataset, i, batch_size, ctr):
    #Placeholder to hold the batch of images
    x = np.zeros((batch_size, img_size, img_size, channels), dtype = np.float32)
    #Placeholder to hold the flatten image batches
    x_flat = np.zeros((batch_size, input_dim), dtype = np.float32)
    for k in range(batch_size):
        x[k] = path_to_numpy(dataset[i+k+ctr])
        x_flat[k] = x[k].flatten()
        #Scaling the images to get to the range of [0,1]
        x_flat[k] = np.multiply(x_flat[k], 1.0/255.0)
    return x_flat

#Function that returns the next image from the dataset provided
def next_image(dataset, i, batch_size):
    #Placeholder to hold the image
    x = np.zeros((1, img_size, img_size, channels), dtype = np.float32)
    #Placeholder to hold the flatten image
    x_flat = np.zeros((1, input_dim), dtype = np.float32)
    x[0] = path_to_numpy(dataset[i+batch_size+1])
    x_flat[0] = x[0].flatten()
    x_flat[0] = np.multiply(x_flat[0], 1.0/255.0)
    return x_flat

