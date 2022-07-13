# Generate a single sample without zeros

import scipy.io as scio #这是v3.7之前的版本需要用的格式
import hdf5storage as hdf
import pandas as pd
import numpy as np
import scipy
import random
import os

import matplotlib.pyplot as plt

# load data
dataPath = './Spatial/Spatial/processed_data/'
real = hdf.loadmat(dataPath+'real_22.mat')
imag = hdf.loadmat(dataPath+'imag_22.mat') # mat文件读入的时候是list类型

real_orig = real['real_pf']
imag_orig = imag['imag_pf']

real = real_orig.reshape([15*300,10240])
imag = imag_orig.reshape([15*300,10240])

# 只取其中前10个分类
real10 = real[0:3000,:]
imag10 = imag[0:3000,:]

num_of_sample = len(real10)