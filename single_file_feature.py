import csv
import os
from pathlib import Path
from vad_mfcc import extract_coef

database_path = Path("/Users/thaipham/Desktop/weekly_project/Depression_Detection_Using_DNN/Depression_Detection_DNN/DATA")
for wav_file in database_path.glob("*.wav"):
  # Extract the .wav filename   
  wav_file = wav_file.stem
  print("## \n Extracting matrix from file: ", wav_file)
      
  # Extract the Participant ID
  participantID = wav_file[0:3]+'_P'
  
  # Get the Matrix
  directory = '/Users/thaipham/Desktop/weekly_project/Depression_Detection_Using_DNN/Depression_Detection_DNN/DATA'
  matrix = extract_coef(directory, wav_file)
  
  # Write the matrix in the csv file
  my_csv_file = directory+'/'+wav_file+'.csv'
  print("## \n Saving matrix as a csv file: ", wav_file+'.csv')
  with open(my_csv_file, 'a') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerows(matrix)
  csvFile.close()