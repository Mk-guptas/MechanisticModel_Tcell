#Defining path for the data
import pathlib
import pandas as pd
import numpy as np



def read_excel(file_path,sheet_name=None):

    data= pd.read_excel(file_path,sheet_name=sheet_name)
    return np.asarray(data["time"]),np.asarray([np.asarray(data["dataset_0"]),np.asarray(data["dataset_1"])])




