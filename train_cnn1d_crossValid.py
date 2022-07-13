
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from time import *

from tensorflow import keras
# from keras.layers import Conv2D, BatchNormalization, Activation, Dense
# from keras.layers import Input, Flatten, AveragePooling2D, Dropout
# from keras import regularizers,layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras import callbacks
from sklearn.model_selection import train_test_split


def load_data(dir_path):
    fopen = open(dir_path, 'r')
    lines = fopen.read().splitlines()  # ????txt
    count = len(open(dir_path, 'r').readlines())  # ??txt????

    data_set = np.empty((count, 1024, 2), dtype="float32")
    label = np.zeros((count), dtype="uint8")

    i = 0
    for line in lines:
        line = line.split(" ")  # ????????

        # img = Image.open(line[0])
        sample = np.load(line[0])
        # print(i, sample.size)
        # img = skimage.io.image(line[0])
        label[i] = int(line[1])

        # img = img.convert('L')          # ?????
        array = np.asarray(sample, dtype="float32")
        data_set[i, :, :] = array

        i += 1

    return data_set, label


# ??CNN??
def model_load(IMG_SHAPE=(1024, 2), class_num=10):
    # ????
    model = tf.keras.models.Sequential([
        # 1D-CNN
        # tf.keras.layers.InputLayer(input_shape=input_shape),
        # tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 10, input_shape=IMG_SHAPE),
        tf.keras.layers.Conv1D(filters=128, kernel_size=21, strides=1, padding='SAME', activation='relu'),
        # tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.BatchNormalization(),
        # ??????, ????,4 ? 3x1 ???, ??? 2
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='SAME'),
        # ??????, 16 ? 23x1 ???
        tf.keras.layers.Conv1D(filters=64, kernel_size=23, strides=1, padding='SAME', activation='relu'),
        # tf.keras.layers.Dropout(rate=0.1),
        # tf.keras.layers.BatchNormalization(),
        # ??????, ????,4 ? 3x1 ???, ??? 2
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='SAME'),
        # ??????, 32 ? 25x1 ???
        tf.keras.layers.Conv1D(filters=32, kernel_size=25, strides=1, padding='SAME', activation='relu'),
        # tf.keras.layers.Dropout(rate=0.1),
        # tf.keras.layers.BatchNormalization(),
        # ??????, ????,4 ? 3x1 ???, ??? 2
        tf.keras.layers.AvgPool1D(pool_size=3, strides=2, padding='SAME'),
        # ??????, 64 ? 27x1 ???
        # tf.keras.layers.Conv1D(filters=64, kernel_size=27, strides=1, padding='SAME', activation='relu'),
        # tf.keras.layers.Dropout(rate=0.5),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Flatten(),
        # The same 128 dense layers, and 10 output layers as in the pre-convolution example:
        tf.keras.layers.Dense(256, activation='relu'),
        # tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        # ??softmax????????????????????????softmax?????
        # tf.keras.layers.Dropout(rate=0.1),
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    # ??????
    model.summary()
    # ??????????????sgd????????????????
    # model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    opt = keras.optimizers.SGD(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # ????
    return model


# ?????????
def show_loss_acc(history, i):
    # ?history??????????????????????
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # ???????????
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(Save_route + 'results_' + "%d" % (i+1) + '.png', dpi=100)


Sample_num_train = 601  # 训练集中每个设备样本的数目
Sample_num_val = 200  # 验证集中每个设备样本的数目
Device_num = 10  # 设备数量
Train_route = 'D:/Code/Polarization/CNN/s1_target/train_label.txt'  # 训练集标签路径
Val_route = 'D:/Code/Polarization/CNN/s1_target/val_label.txt'  # 验证集标签路径
Save_route = 'D:/Code/Polarization/CNN/s1_target/'  # 最佳模型存储位置，图片存储位置


def train(epochs):
    # ???????????
    begin_time = time()
    # todo ?????? ???????????
    x_sample, x_label = load_data(Train_route)  # 训练集路径
    y_sample, y_label = load_data(Val_route)  # 验证集路径
    x = np.zeros(((Sample_num_train+Sample_num_val) * Device_num, 1024, 2), float)
    y = np.zeros(((Sample_num_train+Sample_num_val) * Device_num), int)  # 同上
    x[0:Sample_num_train * Device_num, :, :] = x_sample
    x[Sample_num_train * Device_num:(Sample_num_train+Sample_num_val) * Device_num, :, :] = y_sample
    y[0:Sample_num_train * Device_num] = x_label
    y[Sample_num_train * Device_num:(Sample_num_train+Sample_num_val) * Device_num] = y_label
    index = [i for i in range(len(x[:, 1, 1]))]
    np.random.shuffle(index)
    x = x[index]
    y = y[index]
    y = tf.squeeze(y)
    y = tf.one_hot(y, depth=10)
    print(x.shape)
    x = np.split(x, 10)
    x = np.array(x)
    y = np.split(y, 10)
    y = np.array(y)
    # print(x_label.shape)
    for i in range(10):
        y_sample = x[i]
        y_label = y[i]
        x_label = np.zeros(((Sample_num_train+Sample_num_val) * (Device_num - 1), Device_num), int)
        x_sample = np.zeros(((Sample_num_train+Sample_num_val) * (Device_num - 1), 1024, 2), float)
        for j in range(10):
            if j < i:
                x_sample[j*(Sample_num_train+Sample_num_val):(j+1)*(Sample_num_train+Sample_num_val)] = x[j]
                x_label[j*(Sample_num_train+Sample_num_val):(j+1)*(Sample_num_train+Sample_num_val)] = y[j]
            elif j > i:
                x_sample[(j-1) * (Sample_num_train+Sample_num_val):j * (Sample_num_train+Sample_num_val)] = x[j]
                x_label[(j-1) * (Sample_num_train+Sample_num_val):j * (Sample_num_train+Sample_num_val)] = y[j]
        db_train = tf.data.Dataset.from_tensor_slices((x_sample, x_label))
        db_train = db_train.shuffle(10000)
        db_train = db_train.batch(128)
        db_val = tf.data.Dataset.from_tensor_slices((y_sample, y_label))
        db_val = db_val.shuffle(10000)
        db_val = db_val.batch(128)

        # train_ds, val_ds, class_names = data_load("./slice_data/train",
        # "./slice_data/val", 27, 1024, 16)
        # db_val = tf.data.Dataset.from_tensor_slices((y_sample, y_label))
        # db_val = db_val.shuffle(10000)
        # db_val = db_val.batch(128)

        # ????
        model = model_load()

        # early stopping
        earlystop = EarlyStopping(monitor='val_loss',
                                  min_delta=0.002,
                                  patience=10,
                                  verbose=1)

        mc = ModelCheckpoint(filepath=Save_route + 'best_model' + "%d" % (i+1) + '.h5',
                             monitor='val_accuracy',
                             mode='max',
                             verbose=1,
                             save_best_only=True)

        # ???????epoch?????
        history = model.fit(db_train, batch_size=128, validation_data=db_val, epochs=epochs, verbose=1, callbacks=[mc, earlystop])
        # todo ????? ?????????????
        model.save(Save_route + "cnn_full_mergeModel.h5")
        # ??????
        end_time = time()
        run_time = end_time - begin_time
        print('??????????', run_time, "s")  # ?????????? 1.4201874732
        # ?????????
        show_loss_acc(history, i)


if __name__ == '__main__':
    train(epochs=1500)
