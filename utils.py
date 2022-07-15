from numpy import genfromtxt
import numpy as np
import os
import pandas as pd
from shutil import copy, rmtree
import random

base_url_1 = r'D:\UCLA\LiverImpedance\data\EIS\myDatasets_ex_vivo\normal'

base_url_2 = r'D:\UCLA\LiverImpedance\CNN_Impedance\myDatasets\fatty'

base_url_3 = r'D:\UCLA\LiverImpedance\CNN_Impedance\myDatasets\normal'

def copy_data():
    for file in os.listdir(base_url_1):
        impedance_data = pd.read_csv(os.path.join(base_url_1, file), header=None)
        get_impedance_data = impedance_data.iloc[:, 1:7]
        name_csv = file.split('.')[0].split('_')[2].split('-')[0] + '_' + file.split('.')[0].split('_')[2].split('-')[1] + '.csv'
        name_npy = file.split('.')[0].split('_')[2].split('-')[0] + '_' + file.split('.')[0].split('_')[2].split('-')[1] + '.npy'
        print(name_csv, name_npy)
        get_impedance_data.to_csv(os.path.join(base_url_1, name_csv), index=None, header=None)

def save_numpy():
    for file in os.listdir(base_url_3):
        file_data = genfromtxt(os.path.join(base_url_3, file), delimiter=',')
        file_name = file.split('.')[0] + '.npy'
        np.save(os.path.join(base_url_3, file_name), file_data)

def load_numpy():
    pass

def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)

def split_data():
    # 保证随机可复现
    random.seed(0)

    # 将数据集中10%的数据划分到验证集中
    split_rate = 0.25

    # 指向你解压后的flower_photos文件夹
    cwd = os.getcwd()
    data_root = os.path.join(cwd, "myDatasets")
    origin_impedance_path = os.path.join(data_root, 'liver_impedance')
    assert os.path.exists(origin_impedance_path), "path '{}' does not exist.".format(origin_impedance_path)

    impedance_class = [cla for cla in os.listdir(origin_impedance_path)
                    if os.path.isdir(os.path.join(origin_impedance_path, cla))]

    # 建立保存训练集的文件夹
    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for cla in impedance_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(train_root, cla))

    # 建立保存验证集的文件夹
    val_root = os.path.join(data_root, "val")
    mk_file(val_root)
    for cla in impedance_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(val_root, cla))

    for cla in impedance_class:
        cla_path = os.path.join(origin_impedance_path, cla)
        impedance_data = os.listdir(cla_path)
        num = len(impedance_data)
        # 随机采样验证集的索引
        eval_index = random.sample(impedance_data, k=int(num*split_rate))
        for index, impedance_data in enumerate(impedance_data):
            if impedance_data in eval_index:
                # 将分配至验证集中的文件复制到相应目录
                impedance_data_path = os.path.join(cla_path, impedance_data)
                new_path = os.path.join(val_root, cla)
                copy(impedance_data_path, new_path)
            else:
                # 将分配至训练集中的文件复制到相应目录
                image_path = os.path.join(cla_path, impedance_data)
                new_path = os.path.join(train_root, cla)
                copy(impedance_data_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
        print()

    print("processing done!")

# if __name__ == '__main__':
#     copy_data()
#     save_numpy()
#     split_data()