# -*- coding: utf-8 -*-
# @Time    : 2021/6/17 20:29
# @Author  : dejahu
# @Email   : 1148392984@qq.com
# @File    : train_mobilenet.py
# @Software: PyCharm
# @Brief   : mobilenet模型训练代码，训练的模型会保存在models目录下，折线图会保存在results目录下

import tensorflow as tf
import matplotlib.pyplot as plt
from time import *

class Inception(tf.keras.layers.Layer):
    # 设置模块的构成
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        # 线路1:1*1 RELU same c1
        self.p1_1 = tf.keras.layers.Conv2D(
            c1, kernel_size=1, activation="relu", padding="same")
        # 线路2:1*1 RELU same c2[0]
        self.p2_1 = tf.keras.layers.Conv2D(
            c2[0], kernel_size=1, activation="relu", padding="same")
        # 线路2:3*3 RELU same c2[1]
        self.p2_2 = tf.keras.layers.Conv2D(
            c2[1], kernel_size=3, activation="relu", padding='same')
        # 线路3:1*1 RELU same c3[0]
        self.p3_1 = tf.keras.layers.Conv2D(
            c3[0], kernel_size=1, activation="relu", padding="same")
        # 线路3:5*5 RELU same c3[1]
        self.p3_2 = tf.keras.layers.Conv2D(
            c3[1], kernel_size=5, activation="relu", padding='same')
        # 线路4: max-pool
        self.p4_1 = tf.keras.layers.MaxPool2D(
            pool_size=3, padding="same", strides=1)
        # 线路4:1*1
        self.p4_2 = tf.keras.layers.Conv2D(
            c4, kernel_size=1, activation="relu", padding="same")

    # 前行传播过程
    def call(self, x):
        # 线路1
        p1 = self.p1_1(x)
        # 线路2
        p2 = self.p2_2(self.p2_1(x))
        # 线路3
        p3 = self.p3_2(self.p3_1(x))
        # 线路4
        p4 = self.p4_2(self.p4_1(x))
        # concat
        outputs = tf.concat([p1, p2, p3, p4], axis=-1)
        return outputs


# 数据集加载函数，指明数据集的位置并统一处理为imgheight*imgwidth的大小，同时设置batch
def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    # 加载训练集
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    # 加载测试集
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    # 返回处理之后的训练集、验证集和类名
    return train_ds, val_ds, class_names


# 构建mobilenet模型
# 模型加载，指定图片处理的大小和是否进行迁移学习
def model_load(IMG_SHAPE=(224, 224, 3), class_num=12):
    # 微调的过程中不需要进行归一化的处理
    # 加载预训练的mobilenet模型
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    # 将模型的主干参数进行冻结
    base_model.trainable = False


    model = tf.keras.models.Sequential([
        # 进行归一化的处理
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1, input_shape=IMG_SHAPE),
        # 设置主干模型
        base_model,
        # 对主干模型的输出进行全局平均池化
        tf.keras.layers.GlobalAveragePooling2D(),
        # 通过全连接层映射到最后的分类数目上
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    model.summary()
    # 模型训练的优化器为adam优化器，模型的损失函数为交叉熵损失函数
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 展示训练过程的曲线
def show_loss_acc(history):
    # 从history中提取模型训练集和验证集准确率信息和误差信息
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 按照上下结构将图画输出
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
    plt.savefig('results/results_mobilenet.png', dpi=100)


def train(epochs):
    # 开始训练，记录开始时间
    begin_time = time()
    # todo 加载数据集， 修改为你的数据集的路径
    train_ds, val_ds, class_names = data_load("H:/CNN2/new_data/train",
                                           "H:/CNN2/new_data/val", 224, 224, 16)
    print(class_names)
    # 加载模型
    model = model_load(class_num=len(class_names))
    # 指明训练的轮数epoch，开始训练
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    # todo 保存模型， 修改为你要保存的模型的名称
    model.save("models/mobilenet_picture_video_text.h5")
    # 记录结束时间
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, "s")  # 该循环程序运行时间： 1.4201874732
    # 绘制模型训练过程图
    show_loss_acc(history)


if __name__ == '__main__':
    train(epochs=30)