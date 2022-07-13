# -*- coding: utf-8 -*-
# @Time    : 2021/6/17 20:29
# @Author  : dejahu
# @Email   : 1148392984@qq.com
# @File    : test_model.py
# @Software: PyCharm
# @Brief   : 模型测试代码，测试会生成热力图，热力图会保存在results目录下

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


# 数据加载，分别从训练的数据集的文件夹和测试的文件夹中加载训练集和验证集
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


# 测试mobilenet准确率
def test_mobilenet():
    # todo 加载数据, 修改为你自己的数据集的路径
    # train_ds, test_ds, class_names = data_load("H:/CNN2/new_data/train",
                                              # "H:/CNN2/new_data/val", 224, 224, 16)
    # todo 加载模型，修改为你的模型名称
    model = tf.keras.models.load_model("models/mobilenet_picture_video_text.h5")
    # model.summary()
    # 测试
    loss, accuracy = model.evaluate(test_ds)
    # 输出结果
    print('Mobilenet test accuracy :', accuracy)

    test_real_labels = []
    test_pre_labels = []
    for test_batch_images, test_batch_labels in test_ds:
        test_batch_labels = test_batch_labels.numpy()
        test_batch_pres = model.predict(test_batch_images)
        # print(test_batch_pres)

        test_batch_labels_max = np.argmax(test_batch_labels, axis=1)
        test_batch_pres_max = np.argmax(test_batch_pres, axis=1)
        # print(test_batch_labels_max)
        # print(test_batch_pres_max)
        # 将推理对应的标签取出
        for i in test_batch_labels_max:
            test_real_labels.append(i)

        for i in test_batch_pres_max:
            test_pre_labels.append(i)
        # break

    # print(test_real_labels)
    # print(test_pre_labels)
    class_names_length = len(class_names)
    heat_maps = np.zeros((class_names_length, class_names_length))
    for test_real_label, test_pre_label in zip(test_real_labels, test_pre_labels):
        heat_maps[test_real_label][test_pre_label] = heat_maps[test_real_label][test_pre_label] + 1

    print(heat_maps)
    heat_maps_sum = np.sum(heat_maps, axis=1).reshape(-1, 1)
    # print(heat_maps_sum)
    print()
    heat_maps_float = heat_maps / heat_maps_sum
    print(heat_maps_float)
    # title, x_labels, y_labels, harvest
    show_heatmaps(title="heatmap", x_labels=class_names, y_labels=class_names, harvest=heat_maps_float,
                  save_name="results/heatmap_mobilenet.png")


# 测试cnn模型准确率
def test_cnn():
    # todo 加载数据, 修改为你自己的数据集的路径
    # train_list = './Spatial/train_dataset/train_label.txt'
    val_list = './Spatial/slice_data/29db/test_label.txt'

    # x_sample, x_label = load_data(train_list)
    y_sample, y_label = load_data(val_list)
    # x_label = tf.squeeze(x_label)
    # x_label = tf.one_hot(x_label, depth=10)
    y_label = tf.squeeze(y_label)
    y_label = tf.one_hot(y_label, depth=10)
    # db_train = tf.data.Dataset.from_tensor_slices((x_sample, x_label)).batch(128)
    db_test = tf.data.Dataset.from_tensor_slices((y_sample, y_label)).batch(128)
    class_names = ['d1','d2','d3','d4','d5','d6','d7','d8','d9','d10']
    # todo 加载模型，修改为你的模型名称
    model = tf.keras.models.load_model("./models/cnn_slice_model.h5")
    # model.summary()
    # 测试
    loss, accuracy = model.evaluate(db_test)
    # 输出结果
    print('CNN test accuracy :', accuracy)

    # 对模型分开进行推理
    test_real_labels = []
    test_pre_labels = []
    for test_batch_images, test_batch_labels in db_test:
        test_batch_labels = test_batch_labels.numpy()
        test_batch_pres = model.predict(test_batch_images)
        # print(test_batch_pres)

        test_batch_labels_max = np.argmax(test_batch_labels, axis=1)
        test_batch_pres_max = np.argmax(test_batch_pres, axis=1)
        # print(test_batch_labels_max)
        # print(test_batch_pres_max)
        # 将推理对应的标签取出
        for i in test_batch_labels_max:
            test_real_labels.append(i)

        for i in test_batch_pres_max:
            test_pre_labels.append(i)
        # break

    # print(test_real_labels)
    # print(test_pre_labels)
    class_names_length = len(class_names)
    heat_maps = np.zeros((class_names_length, class_names_length))
    for test_real_label, test_pre_label in zip(test_real_labels, test_pre_labels):
        heat_maps[test_real_label][test_pre_label] = heat_maps[test_real_label][test_pre_label] + 1

    print(heat_maps)
    heat_maps_sum = np.sum(heat_maps, axis=1).reshape(-1, 1)
    # print(heat_maps_sum)
    print()
    heat_maps_float = heat_maps / heat_maps_sum
    print(heat_maps_float)
    # title, x_labels, y_labels, harvest
    show_heatmaps(title="slice 1 heatmap", x_labels=class_names, y_labels=class_names, harvest=heat_maps_float,
                  save_name="./results/spatial/slice_results/29db.png")
    with open('./results/spatial/slice_results/slice_test.txt', 'a+') as f:
        line = '%s\n' % (accuracy)
        f.write(line)




def show_heatmaps(title, x_labels, y_labels, harvest, save_name):
    # 这里是创建一个画布
    fig, ax = plt.subplots()
    # cmap https://blog.csdn.net/ztf312/article/details/102474190
    im = ax.imshow(harvest, cmap="YlGnBu")
    # 这里是修改标签
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(y_labels)
    ax.set_yticklabels(x_labels)

    # 因为x轴的标签太长了，需要旋转一下，更加好看
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 添加每个热力块的具体数值
    # Loop over data dimensions and create text annotations.
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            text = ax.text(j, i, format(harvest[i, j], '.2f'),
                           ha="center", va="center")
    ax.set_xlabel("Predict label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)
    fig.tight_layout()
    plt.colorbar(im)
    plt.savefig(save_name, dpi=100)
    # plt.show()


if __name__ == '__main__':
    # test_mobilenet()
    test_cnn()
