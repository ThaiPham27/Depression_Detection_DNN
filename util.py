import os
import random
from pathlib import Path
# from pathlib import PurePath
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("/Users/thaipham/Desktop/weekly_project/Depression_Detection_Using_DNN/Depression_Detection_DNN/thai_model(14.12).h5",compile = False)

def Load_Data(my_csv_filename):
    # Get Input and Target Data
    input_data = np.ones([1,60])
    i = 1

    input_directory = Path("/Users/thaipham/Desktop/weekly_project/Depression_Detection_Using_DNN/Depression_Detection_DNN/DATA")

    my_csv_filename = Path(my_csv_filename).name

    io = pd.read_csv(str(input_directory)+"/"+my_csv_filename, sep=",", usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60))
    
    # Get the Matrix
    i = i+1
    io = np.array(io, dtype=np.float64)

    # print("###################################################",i)
    
    input_data = np.append(input_data, io, axis=0)                          
    
    return input_data #, target_data_binary_PHQ
    
def prediction():

  input_directory = Path('/Users/thaipham/Desktop/weekly_project/Depression_Detection_Using_DNN/Depression_Detection_DNN/DATA')
  
  dictionary = {}

  for csv_file in input_directory.glob("*.csv"):
    # Extract the .csv filename   
    csv_file_name = csv_file.stem
    
    input_data = Load_Data(csv_file)   

    input_data = np.expand_dims(input_data,-1)
    
    # print(f"Prediction file {csv_file_name}")

    test_predict = model.predict(input_data)

    test_pred1 = np.average(test_predict)

    if test_pred1 >= 0.5:
      label = 'depressed'
    else:
      label = 'non-depressed'

    dictionary[csv_file_name] = [label,round(test_pred1*100,2)]
  
  return dictionary
