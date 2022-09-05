# 生成图像及标签文件https://blog.csdn.net/u010682375/article/details/77746489
import os
from fnmatch import fnmatchcase as match


def generate(dir, label):
    files = os.listdir(dir)
    files.sort()
    print('start...')
    listText = []
    i = 0
    for i in range(10):
        listText.insert(i, open(dir + 'label_list' + "%d.txt" % (i + 1), 'w'))
    for file in files:
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        name = file + ' ' + str(int(label)) + '\n'
        if match(file, 's1_*'):
            listText[0].write(dir + name)
        elif match(file, 's2_*'):
            listText[1].write(dir + name)
        elif match(file, 's3_*'):
            listText[2].write(dir + name)
        elif match(file, 's4_*'):
            listText[3].write(dir + name)
        elif match(file, 's5_*'):
            listText[4].write(dir + name)
        elif match(file, 's6_*'):
            listText[5].write(dir + name)
        elif match(file, 's7_*'):
            listText[6].write(dir + name)
        elif match(file, 's8_*'):
            listText[7].write(dir + name)
        elif match(file, 's9_*'):
            listText[8].write(dir + name)
        elif match(file, 's10_*'):
            listText[9].write(dir + name)
    i = 0
    for i in range(10):
        listText[i].close()
    print('down!')


SNR_num = '0'

if __name__ == '__main__':
    PATH = '/Volumes/Seagate/graduate/data-backup/Spatial/slice_data/' + SNR_num + 'db/'
    generate(PATH + 'd1/', 0)
    generate(PATH + 'd2/', 1)
    generate(PATH + 'd3/', 2)
    generate(PATH + 'd4/', 3)
    generate(PATH + 'd5/', 4)
    generate(PATH + 'd6/', 5)
    generate(PATH + 'd7/', 6)
    generate(PATH + 'd8/', 7)
    generate(PATH + 'd9/', 8)
    generate(PATH + 'd10/', 9)
    # generate(PATH+'d11/', 10)
    # generate(PATH+'d12/', 11)
    # generate(PATH+'d13/', 12)
    # generate(PATH+'d14/', 13)
    # generate(PATH+'d15/', 14)
