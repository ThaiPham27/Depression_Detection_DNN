from flask import Flask, render_template, request, redirect, url_for
import os
import random
import pathlib as Path
from pathlib import Path
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
import csv
from tensorflow.keras.models import load_model
from vad_mfcc import extract_coef
app = Flask(__name__, static_folder="static")

# model
model = load_model("./thai_model(14.12).h5",compile = False)


# load data from csv file
def Load_Data(my_csv_filename):
    # Get Input and Target Data
    input_data = np.ones([1,60])
    i = 1

    input_directory = ("/Users/thaipham/Desktop/weekly_project/Depression_Detection_Using_DNN/Depression_Detection_DNN/static")

    my_csv_filename = Path(my_csv_filename).name

    io = pd.read_csv(input_directory+"/"+my_csv_filename, sep=",", usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60))
    
    # Get the Matrix
    i = i+1
    io = np.array(io, dtype=np.float64)

    # print("###################################################",i)
    
    input_data = np.append(input_data, io, axis=0)                          
    
    return input_data #, target_data_binary_PHQ

# make prediction
def prediction():

  input_directory = Path('/Users/thaipham/Desktop/weekly_project/Depression_Detection_Using_DNN/Depression_Detection_DNN/static')
  
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
    break
  return dictionary

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    file = request.files['file']
    filename = file.filename
    file.save("/Users/thaipham/Desktop/weekly_project/Depression_Detection_Using_DNN/Depression_Detection_DNN/static/"+filename)
    
    # extract features to csv files
    database_path = Path("/Users/thaipham/Desktop/weekly_project/Depression_Detection_Using_DNN/Depression_Detection_DNN/static")
    
    for wav_file in database_path.glob("*.wav"):
        # Extract the .wav filename   
        wav_file = wav_file.stem
        print("## \n Extracting matrix from file: ", wav_file)
        
        # Get the Matrix
        directory = '/Users/thaipham/Desktop/weekly_project/Depression_Detection_Using_DNN/Depression_Detection_DNN/static'
        matrix = extract_coef(directory, wav_file)
        
        # Write the matrix in the csv file
        my_csv_file = directory+'/'+wav_file+'.csv'
        print("## \n Saving matrix as a csv file: ", wav_file+'.csv')
        with open(my_csv_file, 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(matrix)
        csvFile.close()

        break
    
    fname = filename
    # print(fname)

    # make prediction
    result = prediction()
    # print(result)
    return render_template('predict.html', result=result, fname=fname)


if __name__ == '__main__':
    app.run(debug=True)

    # # Serve the app with gevent
    # http_server = WSGIServer(('', 5000), app)
    # http_server.serve_forever()