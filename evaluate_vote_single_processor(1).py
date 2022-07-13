import tensorflow as tf
import numpy as np


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


def find_max(x):  # 返回数组中最大值的下标
    m = 0
    max_num = max(x)
    max_index = 0
    for m in range(len(x)):
        if x[m] == max_num:
            max_index = m
    return max_index


def get_weight():
    acc = np.zeros((len(Subnetwork_index), SNR_num), float)  # 用于存储每个子网络的验证准确率
    loss = np.zeros((len(Subnetwork_index), SNR_num), float)  # 用于存储每个子网络的损失函数，备用
    for i in Subnetwork_index:  # 计算每个子网络不同信噪比下的验证准确率作为投票权重
        model = tf.keras.models.load_model(model_path + 'best_model' + ".h5" # 选择每个子网络最优模型输入
        for j in range(1):  # 获取每个信噪比档位下的准确率
            for k in range(Device_num):
                val_list = data_path + "30db/"+ 'd%d/' % (k + 1) +'label_list' + "%d.txt" % (i + 1)
                if j == 30:
                    val_list = data_path + 'pfdb/' + 'd%d/' % (k + 1) + 'label_list' + "%d.txt" % (i + 1)
                y_sample, y_label = load_data(val_list)
                y_label = tf.squeeze(y_label)
                y_label = tf.one_hot(y_label, depth=10)
                temp1, temp2 = model.evaluate(y_sample, y_label, batch_size=32)
                loss[i][j] = temp1 + loss[i][j]
                acc[i][j] = temp2 + acc[i][j]
            loss[i][j] = loss[i][j] / Device_num
            acc[i][j] = acc[i][j] / Device_num
            print(j)
        print(i / len(Subnetwork_index), '1')
    print(acc)  # 输出子网络准确率矩阵（频段数量*信噪比档位数）
    return acc


def get_pre_mat():
    y_pre = np.zeros((len(Subnetwork_index), SNR_num, Sample_num * Device_num, Device_num), float)  # 预测结果矩阵
    y_tru = np.zeros((SNR_num, Sample_num * Device_num), int)  # 实际结果矩阵
    for i in Subnetwork_index:  # 获得每个子网络针对不同信噪比的预测结果
        model = tf.keras.models.load_model(model_path + 'best_model' + "%d.h5" % (i + 1))
        for j in range(SNR_num):  # 获得某一子网络不同信噪比下的预测结果和真实结果
            for k in range(Device_num):
                val_list = data_path + "%ddb/" % (j) + 'd%d/' % (k + 1) +'label_list' + "%d.txt" % (i + 1)
                if j == 30:
                    val_list = data_path + 'pfdb/' + 'd%d/' % (k + 1) + 'label_list' + "%d.txt" % (i + 1)
                y_sample, y_label = load_data(val_list)
                y_label = tf.squeeze(y_label)
                y_tru[j][(Sample_num * k):(Sample_num * (k+1))] = y_label[0:Sample_num]
                y_pre[i][j][(Sample_num * k):(Sample_num * (k+1))] = model(y_sample, training=False)[:][:][0:Sample_num]
            print(j / SNR_num, i / len(Subnetwork_index), '2')
    return y_pre, y_tru


def get_vote(y_pre, acc):
    results = np.zeros((SNR_num, Sample_num * Device_num), float)  # 投票结果矩阵，信噪比档位数*每个信噪比下样本数
    result_pre = np.zeros((Device_num), float)
    for j in range(SNR_num):  # 获得投票结果,结果保存在results中，第一层对SNR循环
        for k in range(Sample_num * Device_num):  # 计算每一个样本的判决结果
            for i in Subnetwork_index:  # 对子网络循环记录判决结果
                for l in range(Device_num):  # 获得每个设备的认定概率=子网络判决概率*子网络在当前信噪比下准确率
                    result_pre[l] += y_pre[i][j][k][l] * acc[i][j]
            results[j][k] = find_max(result_pre)
            result_pre = np.zeros((Device_num), float)
        print(j / SNR_num, '3')
    return results


def get_final(results, y_tru):
    Right = 0
    Wrong = 0
    Accuracy_fi = np.zeros((SNR_num))  # 最终结果，每个信噪比一个准确率
    for j in range(SNR_num):  # 对比投票结果和真实结果，获取每个信噪比条件下投票系统准确率
        for k in range(Sample_num * Device_num):
            if results[j][k] == y_tru[j][k]:
                Right += 1
            else:
                Wrong += 1
        Accuracy_fi[j] = Right / (Right + Wrong)  # 一个信噪比档位下所有样本计算完成后获得一个准确率结果
        Right = 0
        Wrong = 0
        #  print(Accuracy_fi[j])
        print(j/SNR_num, '4')
    return Accuracy_fi


SNR_num = 1  # 共有多少个信噪比档位
Subnetwork_index = [0, 1, 2, 3, 7, 8, 9]  # 所使用的子网络序号，也可以改成随机实现
Sample_num = 300  # 每个标签文件中样本的数目
Device_num = 10  # 设备数量
data_path = './Spatial/single_slice_data/'
model_path = './Spatial/models/'

if __name__ == '__main__':

    accuracy_mat = get_weight()
    y_predict, y_truth = get_pre_mat()
    result_matrix = get_vote(y_predict, accuracy_mat)
    final_result = get_final(result_matrix,y_truth)

    print("投票系统运算准确率为：\n")
    print(final_result)
