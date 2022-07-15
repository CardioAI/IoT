from numpy import genfromtxt
import numpy as np
import pandas as pd
import os


for file in os.listdir(r'D:\UCLA\LiverImpedance\CNN_Impedance\myDatasets\liver_impedance'):
    df = pd.read_csv(r'D:\UCLA\LiverImpedance\CNN_Impedance\myDatasets\liver_impedance' + '/' + file, header=None)
    np.save('./myDatasets/' + file.split('.')[0] + '.npy', df)
    print(df.shape)

