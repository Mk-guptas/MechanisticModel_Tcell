#Defining path for the data
import pathlib
import pandas as pd
import numpy as np



def read_excel(file_path,sheet_name=None):

    data= pd.read_excel(file_path,sheet_name=sheet_name)
    return np.asarray(data["time"]),np.asarray([np.asarray(data["dataset_0"]),np.asarray(data["dataset_1"])])

#time_data,population=read_excel(exp_path+'/tumor_cell_data.xlsx',sheet_name="coated")
#print(np.shape(time_data));
#print(np.shape(population))




