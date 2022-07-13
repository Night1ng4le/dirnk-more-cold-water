import scipy.io as scio #这是v3.7之前的版本需要用的格式
import hdf5storage as hdf
import pandas as pd
import numpy as np
import scipy
import random
import os
import multiprocessing
from sklearn import preprocessing

import matplotlib.pyplot as plt

# get all the files of the path,读取文件夹下所有文件名字到一个列表
def readname(filePath):
    # 这里一定要使用绝对路径
    name = os.listdir(filePath)
    return name

def gen(folder_path, save_PATH, file_names, folder):

    for file_name in file_names:
#        print(file_name)
        name, suffix = os.path.splitext(file_name)
        # print(name)
        src_path = folder_path + '/' + file_name
        # print(src_path)
        # save_path = des_PATH + folder +name
        save_path = save_PATH + folder + '/' + name + '.png'
        # print(save_path)
        array = np.load(src_path, allow_pickle=True)
        # # print(arr)
        # # plt.figure()
        array_normal = min_max_scaler.fit_transform(array)
        colors = array_normal[:,1]
        plt.scatter(array_normal[:,0],array_normal[:,1],marker = 'o',alpha = 0.5, c = colors,cmap='viridis')
        plt.savefig(save_path)
        # plt.show()
        plt.close()
if __name__ == '__main__':
    # 读取每个类别下的完整样本
    filePath = '/Volumes/Seagate/graduate/data-backup/Spatial/slice_data/4db/'
    folder_names = readname(filePath) #这里一定要使用绝对路径
    # del folder_names[0]
    print(folder_names)

    # 循环生成图片样本
    save_PATH = '/Users/hurrican/Desktop/experiment/Spatial/slice_png/30db/'
    folder_name = folder_names[1:6]+folder_names[7:]
    min_max_scaler = preprocessing.MinMaxScaler()
    threads = []
    i = 0
    for folder in folder_name:
        # print(folder)
        folder_path = filePath + folder
        # print(folder_path)
        file_names = readname(folder_path)
        thread_new = multiprocessing.Process(target=gen, args=(folder_path, save_PATH, file_names, folder))
        threads.append(thread_new)
        threads[i].start()
        i += 1

    for thread in threads:
        thread.join()


    print('over')
