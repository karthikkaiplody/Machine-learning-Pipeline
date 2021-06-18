# Standard helper functions for structured data

# importing the libraries
# Standard
import numpy as np
import pandas as pd

# Pycaret
from pycaret.classification import *

# Plots
from plotly.offline import iplot
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns


# Getting the dataset info
def dataset_info(dataset, dataset_name:str ):
    print(f"Dataser Name:{dataset_name} | Number of samples: {dataset.shape[0]} | Number of columns: {dataset.shape[1]}")
    print(30*"=")
    print("Column           Data Type")
    print(dataset.dtypes)
    print(30*"=")
    missing_data = dataset.isnull().sum()
    if sum(missing_data) > 0:
        print(missing_data[missing_data.data.values > 0])
    else:
        print("Looks like there is no missing Data!!")
    print(30*"=")
    print(f"Memory Usage: {np.round(dataset.memory_usage(index=True).sum() / 10e5, 3)} MB") 

# Data sampling
def data_sampling(dataset, frac:float, random_seed: int):
    data_sampled_a = dataset.sample(frac=frac, random_state=random_seed)
    data_sampled_b = dataset.drop(data_sampled_a.index).reset_index(drop=True)
    data_sampled_a.reset_index(drop=True, inplace=True)
    return data_sampled_a, data_sampled_b