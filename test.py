# from flask import Flask, render_template, request, redirect
import os
import random
from pathlib import Path
# from pathlib import PurePath
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

for root, dirs, files in os.walk("/Users/thaipham/Desktop/weekly_project/Depression_Detection_Using_DNN/Depression_Detection_DNN/DATA", topdown=False):
    # print(root)
    # print(dirs)
    print(files)