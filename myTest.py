import numpy as np
import os
import re

name_list = []
for np_file in os.listdir(r'D:\UCLA\LiverImpedance\CNN_Impedance\myDatasets\all_data'):
    name = np_file.split('_')[0]
    name_list.append(name)

name_list = list(set(name_list))

for name in name_list:
    name_np = np.zeros((30, 6, 1), dtype=float)
    for np_file in os.listdir(r'D:\UCLA\LiverImpedance\CNN_Impedance\myDatasets\all_data'):
        if np_file.split('_')[0] == name:
            np_data = np.load(os.path.join(r'D:\UCLA\LiverImpedance\CNN_Impedance\myDatasets\all_data', np_file))
            name_np = np.dstack((name_np, np_data))

    name_np = np.delete(name_np, 0, 2)
    print(name_np.shape)
    np.save(os.path.join(r'D:\UCLA\LiverImpedance\CNN_Impedance\myDatasets\all_data', name), name_np)









