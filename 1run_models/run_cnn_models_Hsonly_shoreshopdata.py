
from utils.splits import normal_data, split_sequences
from utils.loss_errors import index_mielke, mielke_loss

import os
import random
import numpy as np
import pandas as pd
import math
import scipy.stats

from datetime import  timedelta
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D


abs_path= os.path.abspath(os.getcwd())
os.chdir(abs_path)


##Import Observed Shoreline positions
mat_in= pd.read_csv('../data/shoreshop2_shorelines_obs.csv') 


