# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 16:08:16 2017

@author: ziken
"""
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

VALIDATION_SIZE = 100       # size of validation set
EPOCHS = 100                # epochs count
BATCH_SIZE = 64
EARLY_STOP_PATIENCE = 10    # if error increases continuously for EARLY_STOP_PATIENCE iterations, stop training

'''
    X.shape: (2140, 96, 96, 1)   2140 fotos of size 96x96, 1 channel
    label.shape: (2140, 30)      each foto has 15 key_points, each key_point has (x, y)
'''
def load_training_data(training_file):    
    df = pd.read_csv(training_file)
    
    # columns[:-1] : label y(s);
    # the last colume is raw 'image'. exclude it.
    label_cols = df.columns[:-1] 
    print('label_cols (no image):', label_cols)
    
    # drop all incompleted samples. only 2140 completed samples are left.
    # *** should do some feature engineering here to make up those incompleted samples ***
    df = df.dropna()
    print(df.head())
    
    # df['Image'] is the last column (input x)
    # stratch the pixel values of image(input x) to [0, 1]
    df['Image'] = df['Image'].apply(lambda pixel: np.fromstring(pixel, sep=' ') / 255.0)
    
    # --------- reshape the raw image from 9216 to 96x96x1 ---------
    #   2140 valid fotos in total. each foto has 9216 pixels.
    #   df[Image].shape: (2140,),   df[Image][0].shape: (9216,)
    #   transfer it into ndarray of shape(2140, 9216) , then reshape to (2140, 96, 96, 1)
    X = np.vstack(df['Image'])      
    X = X.reshape((-1, 96, 96, 1)) 

    # labels y(s), which are the all columns but the last one
    # y.shape = (2140, 30)   15 keypoints each face with x, y for each keypoint
    labels = df[label_cols].values / 96.0       #将label (y)值缩放到[0, 1]区间
    print('labels.shape: ', labels.shape) 

    # X.shape: (2140, 96, 96, 1)
    # y.shape: (2140, 30)
    return X, labels

def load_testing_data(testing_file):
    df = pd.read_csv(testing_file)
    df['Image'] = df['Image'].apply(lambda pixel: np.fromstring(pixel, sep=' ') / 255.0)
    X = np.vstack(df['Image'])
    X = X.reshape((-1, 96, 96, 1))
    return X



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

# rmse(root mean square error): loss function
# output_conv = f(x)
def build_nn_model(x, y_, keep_prob):
    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([2, 2, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([2, 2, 64, 128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    W_fc1 = weight_variable([11 * 11 * 128, 500])
    b_fc1 = bias_variable([500])

    h_pool3_flat = tf.reshape(h_pool3, [-1, 11 * 11 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([500, 500])
    b_fc2 = bias_variable([500])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    W_fc3 = weight_variable([500, 30])
    b_fc3 = bias_variable([30])

    predict_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_ - predict_conv)))
    return predict_conv, rmse    



keypoint_index = {
    'left_eye_center_x':0,
    'left_eye_center_y':1,
    'right_eye_center_x':2,
    'right_eye_center_y':3,
    'left_eye_inner_corner_x':4,
    'left_eye_inner_corner_y':5,
    'left_eye_outer_corner_x':6,
    'left_eye_outer_corner_y':7,
    'right_eye_inner_corner_x':8,
    'right_eye_inner_corner_y':9,
    'right_eye_outer_corner_x':10,
    'right_eye_outer_corner_y':11,
    'left_eyebrow_inner_end_x':12,
    'left_eyebrow_inner_end_y':13,
    'left_eyebrow_outer_end_x':14,
    'left_eyebrow_outer_end_y':15,
    'right_eyebrow_inner_end_x':16,
    'right_eyebrow_inner_end_y':17,
    'right_eyebrow_outer_end_x':18,
    'right_eyebrow_outer_end_y':19,
    'nose_tip_x':20,
    'nose_tip_y':21,
    'mouth_left_corner_x':22,
    'mouth_left_corner_y':23,
    'mouth_right_corner_x':24,
    'mouth_right_corner_y':25,
    'mouth_center_top_lip_x':26,
    'mouth_center_top_lip_y':27,
    'mouth_center_bottom_lip_x':28,
    'mouth_center_bottom_lip_y':29
}

if __name__ == '__main__':
    sess = tf.InteractiveSession()

    # use placeholds to build network layers
    x = tf.placeholder("float", shape = [None, 96, 96, 1])
    y_ = tf.placeholder("float", shape = [None, 30])
    keep_prob = tf.placeholder("float")
    predict_conv, rmse = build_nn_model(x, y_, keep_prob)
    train_step = tf.train.AdamOptimizer(1e-3).minimize(rmse)

    print('initialize all variables...')    
    init = tf.global_variables_initializer()
    sess.run(init)

    # --------- load samples from training set and build model ----------
    print('load all 2140 training + validation samples (with labels).')
    X, labels = load_training_data(os.getcwd() + '/training.csv')
    
    print('split them into training_set and validation_set (2140-100 : 100)')
    X_train, y_train = X[VALIDATION_SIZE:], labels[VALIDATION_SIZE:]
    X_valid, y_valid = X[:VALIDATION_SIZE], labels[:VALIDATION_SIZE]
    TRAIN_SIZE = X_train.shape[0]

    print('begin training..., train dataset size:{0}'.format(TRAIN_SIZE))
    best_validation_loss = 1000000.0
    current_epoch = 0
    for i in range(EPOCHS):                                     # each epoch
        # shuffle the indices of training samples
        TRAIN_SIZE = X_train.shape[0]
        train_index = list(range(0, TRAIN_SIZE, 1))
        random.shuffle(train_index)
        X_train, y_train = X_train[train_index], y_train[train_index]

        for j in tqdm(range(0, TRAIN_SIZE, BATCH_SIZE)):        # each batch
            # print('epoch {0}, train {1} samples done...'.format(i, j))
            train_step.run(feed_dict = {x : X_train[j : j + BATCH_SIZE], \
                                       y_ : y_train[j : j + BATCH_SIZE], keep_prob : 0.5})

        train_loss = rmse.eval(feed_dict = {x : X_train, y_ : y_train, keep_prob : 1.0})
        validation_loss = rmse.eval(feed_dict={x : X_valid, y_ : y_valid, keep_prob : 1.0})

        print('epoch {0} done! training loss:{1}; validation loss:{2}'.format(i, train_loss*96.0, validation_loss*96.0))
        
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            current_epoch = i
            saver = tf.train.Saver()
            model_saved = saver.save(sess, os.getcwd() + '/model')
            print('model saved in :{0}'.format(model_saved))            
        elif (i - current_epoch) >= EARLY_STOP_PATIENCE:
            print('early stopping')
            break
    # --------- load training set 'training.csv' and build model ----------


    # --------- load test set 'test.csv' to predict key_points  ----------
    X = load_testing_data(os.getcwd() + '/test.csv')
    predict_output = []

    TEST_SIZE = X.shape[0]
    for j in range(0, TEST_SIZE, BATCH_SIZE):
        y_batch = predict_conv.eval(feed_dict={x : X[j : j + BATCH_SIZE], keep_prob : 1.0})
        predict_output.extend(y_batch)
    print('predict test image done!')
    # --------- load test set 'test.csv' to predict key_points  ----------


    resultfile = open('./result.csv','w')
    resultfile.write('RowId, ImageId, FeatureName,Location\n')
    submitfile = open('./submit.csv', 'w')
    submitfile.write('RowId,Location\n')

    IdLookupTable = open('IdLookupTable.csv')
    IdLookupTable.readline()

    for line in IdLookupTable:
        RowId,ImageId,FeatureName = line.rstrip().split(',')
        image_index = int(ImageId) - 1
        feature_index = keypoint_index[FeatureName]
        feature_location = predict_output[image_index][feature_index] * 96
        resultfile.write('{0},{1},{2},{3}\n'.format(RowId, ImageId, FeatureName, feature_location))
        submitfile.write('{0},{1}\n'.format(RowId, feature_location))

    resultfile.close()
    submitfile.close()
    IdLookupTable.close()
