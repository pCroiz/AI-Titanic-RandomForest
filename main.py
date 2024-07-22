import tensorflow as tf
from keras import layers
import pandas as pd
import numpy as np
import os

### Get the data ###
path = os.getcwd()
dataTrain = pd.read_csv(path + '/data/train.csv')
dataTest = pd.read_csv(path + '/data/test.csv')

### Preprocess the data ###
print(dataTrain)

def preprocess(df):
    df = df.copy()
    
    def normalize_name(x):
        return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])
    
    def ticket_number(x):
        return x.split(" ")[-1]
        
    def ticket_item(x):
        items = x.split(" ")
        if len(items) == 1:
            return "NONE"
        return "_".join(items[0:-1])
    
    df["Name"] = df["Name"].apply(normalize_name)
    df["Ticket_number"] = df["Ticket"].apply(ticket_number)
    df["Ticket_item"] = df["Ticket"].apply(ticket_item)   

    # Error woth the Cabin column 
    df.drop('Cabin', axis=1, inplace=True)

    # Error with the Embarked column
    df.drop('Embarked', axis=1, inplace=True)       

    return df
    
preProc_dataTrain = preprocess(dataTrain)
preProc_dataTest = preprocess(dataTest)

### Get the input features ###

features = []
removesFeature = ["Ticket","PassengerId","Survived"]

for key in preProc_dataTrain.keys():
    if key != 'Survived':
        features.append(layers.Input(shape=(1,), name=key))

### Definition of the Input Function ###

# Usefull to convert the pd dataset to an tf dataset
def input_fn(data, labels=None, training=True, batch_size=32):
    # Create a tf.data.Dataset object from the input data and labels
    dataset = tf.data.Dataset.from_tensor_slices((dict(data), labels))

    # If training is True, shuffle the data and repeat it indefinitely
    if training:
        dataset = dataset.shuffle(1000).repeat()

    # Return the batched dataset
    return dataset.batch(batch_size)
