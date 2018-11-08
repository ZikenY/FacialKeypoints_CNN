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
