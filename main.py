import torch
import pandas as pd
import os

# Get the data
path = os.getcwd()
dataTrain = pd.read_csv(path + '/data/train.csv')
dataTest = pd.read_csv(path + '/data/test.csv')



