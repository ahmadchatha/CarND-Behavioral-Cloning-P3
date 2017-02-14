import csv
import os
import cv2

from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers.core import Lambda
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data_path = 'data2/driving_log.csv'
udacity_path = 'data/driving_log.csv'

images_names = []
angles = []

with open(data_path, 'r') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',')
	for row in spamreader:
		images_names.append(row[0])
		angles.append(row[3])


print(images_names[0])
print(angles[0])
print(len(images_names) == len(angles))
