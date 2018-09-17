# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 16:05:33 2017

@author: ziken
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import os

cwd = os.getcwd()

test_file = cwd + '/test.csv'
result_file = cwd + '/result.csv'

df = pd.read_csv(test_file)
df['Image'] = df['Image'].apply(lambda img: np.fromstring(img, sep=' ') / 255.0)
test = np.vstack(df['Image'])
#print(test.shape)  #(1783, 9216)
test = test.reshape((-1,96,96,1)) #   (1783, 96, 96, 1)

df = pd.read_csv(result_file)
locations = np.vstack(df['Location'])

j = 0
#for i in range(0, test.shape[0]-1):
for i in range(0, 10):
    z = test[i, ...]   # get the first dimension (96, 96, 1)
    
    z.resize((z.shape[0], z.shape[1]))  #(96, 96)
    
    plt.figure(i)
    plt.imshow(z, cmap = cm.Greys_r)
    
    for k in range(0, 15):
        plt.plot(locations[j], locations[j+1], marker='x')
        j = j+2
    plt.show()

        
    
'''
def show_barbie(jpg_file):
    im = plt.imread(jpg_file, format = 'jpeg')
    plt.imshow(im, cmap = cm.Greys_r)
    im4d = np.expand_dims(im, axis=0)

    model = tf.Graph()
    with model.as_default():
        # perform a 1x1 convolution
        # shape = 1 x 1 x 3 x 1
        wts      = tf.constant([[[[0.21], [0.72], [0.07]]]],    # convolution kernel
                               dtype=tf.float32)                # shape=(1, 1, 3, 1)
        #input
        my_image = tf.constant(im4d, dtype=tf.float32)
        
        # gray is a op executing 1x1 convolution
        gray     = tf.nn.conv2d(my_image, wts, [1, 1, 1, 1], padding='SAME')
        
    # use 'model' as new graph to execute gray operation
    with tf.Session(graph=model) as sess:
        output = sess.run(gray)

    output.resize((z.shape[0], z.shape[1]))  #(96, 96)
    plt.figure(jpg_file)
    plt.imshow(output, cmap = cm.Greys_r)
'''
