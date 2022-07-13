### dataset说明

- data - 之前的分体数据集

- Spatial - 预处理之后的原始数据

- slice_data - 10段混合分段切片数据

- 30results.txt - 一段0～30db识别结果

- merge_dataset - 不分段混合数据集（zero_vector的混合数据集，一共10000条数据）6.20 

- non_zero_vector - 完整数据样本切片（原始数据切片）6.17

- zero_vector - 随机缺失段数据（每类设备每种情况100条）6.20

- train_dataset - merge_dataset按8:2切分后的训练数据 6.20

- test_dataset - 0～29db不同信噪比构成的测试集 6.21

- single_slice_data - 10段分开的单段切片数据，用于组合和子网络训练

- slice_train - 单段切片10段混合训练数据


### slice_data工作流程
- slice_data
    - 0~30db - 不同信噪比在不同文件夹
        - s1~s10 - 表示十段
            - d1～d10 - 10个设备
            
只对s1使用split_data.py切分30db下的数据，8:2 - 训练集：验证集
- train
    - d1～d10
- val
    - d1~d10
使用label分别对train和val文件夹下的样本生成标签
（这里需要把标签手动合成到一个txt，然后放在和train、val同一级的文件夹下

然后就能训练了
