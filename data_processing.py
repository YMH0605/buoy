import numpy as np
import json
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime
from datetime import timedelta
import firebase_admin
from firebase_admin import db
from firebase_admin import credentials
import pytz
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily, Hourly, Stations
from sklearn.model_selection import KFold
import pvlib
import urllib.request
import time
from pvlib.location import Location
import pytorch_lightning as pl
import random
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from scipy.signal import savgol_filter
from sklearn.inspection import permutation_importance
import copy

from astral import LocationInfo
from astral.sun import sun
from datetime import date

pl.seed_everything(42)
random.seed(42)
torch.manual_seed(42)
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    firebase_admin.delete_app(app)
except:
    print('making new app')
cred = credentials.Certificate("fb_key.json")
app = firebase_admin.initialize_app(cred, {'databaseURL': 'https://haucs-monitoring-default-rtdb.firebaseio.com'})

# Remove outliers in DO and temperature lists
def outliers(lst):
    lst = np.array(lst)
    data = [x for x in lst if isinstance(x, (int, float, np.number, np.ndarray))]
    if len(data) == 0:
        return np.nan
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    return filtered_data

def remove_outliers_from_df(df):
    for column in df.columns:
        df[column] = df[column].apply(outliers)
    return df

# Calculate the weighted mean of DO and temp
def calculate_weighted_mean(lst):
    if not isinstance(lst, list):
        return lst
    num_double_weights = min(len(lst), 6)
    weights = [1] * (len(lst) - num_double_weights) + [2] * num_double_weights
    return np.average(lst, weights=weights)

def Convert_mean(df):
    for column in df.columns:
        df[column] = df[column].apply(calculate_weighted_mean)
    return df

def denormalize(predictions, mean, std):
    return predictions * std + mean


def norimalize(data, mean, std):
    return (data - mean) / std

# Introduce time as another feature
def fix_time(data):
    datetime_object= pd.to_datetime(data.index, format='%Y%m%d_%H:%M:%S')
  # print(datetime_object)
    datetime_object= datetime_object.round('60min').to_pydatetime()
  # print(data['datetime'])
  # data['year'] =  pd.DatetimeIndex(data['datetime']).year
  # data['month'] =pd.DatetimeIndex(data['datetime']).month
  # data['day'] =pd.DatetimeIndex(data['datetime']).day
  # data['hour'] = pd.DatetimeIndex(datetime_object).hour
  # data.set_index('datetime', inplace = True)
    return data

def convert_to_mgl(do_input, t, p, s=0):
    T = t + 273.15; #temperature in kelvin
    P = p * 9.869233e-4; #pressure in atm

    DO_baseline = math.exp(-139.34411 + 1.575701e5/T - 6.642308e7/math.pow(T, 2) + 1.2438e10/math.pow(T, 3) - 8.621949e11/math.pow(T, 4))

    Fs = math.exp(-s * (0.017674 - 10.754/T + 2140.7/math.pow(T, 2)))

    theta = 0.000975 - 1.426e-5 * t + 6.436e-8 * math.pow(t, 2)
    u = math.exp(11.8571 - 3840.7/T - 216961/math.pow(T, 2))
    Fp = (P - u) * (1 - theta * P) / (1 - u) / (1 - theta)

    DO_corrected = DO_baseline * Fs * Fp
    DO_mgl = do_input / 100 * DO_corrected

    return DO_mgl


# Incorportae all functions above, get processed temp and do data wth pond_id
# Incorportae all functions above, get processed temp and do data wth pond_id
def getData_5feature_tensor(pond_number):
    base_path = '/LH_Farm/pond_'
    pond = f"{base_path}{pond_number}"  #Get data from the firebase
    ref = db.reference(pond)
    data = ref.get()
    df = pd.DataFrame(data)
    df = df.T
    
    df = df[df['type'] == 'a_buoy']
    df = df[df.index.str[:8] > '20250701']
    df_do_related = df[['do', 'temp', 'pressure']]
    df_init_do = df[['init_do']]

    #do_res = remove_outliers_from_df(df_do_related)
    do_mean = Convert_mean(df_do_related) # Get mean of DO
    do_mean = pd.concat([do_mean, df_init_do], axis=1)
    do_mean['do'] = do_mean.apply(lambda row: convert_to_mgl(100 * row['do'] / row['init_do'], row['temp'], row['pressure']), axis=1)

    do_mean['datetime'] = pd.to_datetime(do_mean.index, format='%Y%m%d_%H:%M:%S')

    do_mean['datetime'] = do_mean['datetime'].dt.tz_localize('UTC')
    do_mean['datetime'] = do_mean['datetime'].dt.tz_convert('America/Chicago')

    do_mean['hour'] = do_mean['datetime'].dt.hour
    do_mean['minute'] = do_mean['datetime'].dt.minute
    do_mean['hour_minute'] = do_mean['hour'] + do_mean['minute'] / 60.0

    do_mean['formatted_index'] = do_mean['datetime'].dt.strftime('%Y%m%d_%H:%M:%S')
    do_mean = do_mean.set_index('formatted_index', drop=True)
    do_mean = do_mean.drop('datetime', axis = 1)
    do_mean = do_mean.drop('hour', axis = 1)
    do_mean = do_mean.drop('minute', axis = 1)
    do_mean = do_mean.iloc[:-10]
    do_mean['datetime'] = pd.to_datetime(do_mean.index, format='%Y%m%d_%H:%M:%S')

    do_mean_index = do_mean.index


    do_mean = do_mean[['do', 'temp', 'hour_minute']]
    do_mean = do_mean.dropna()
    with open('latest_data.json', 'w') as f:
        json.dump(data, f)

    return do_mean


# Get data from all six ponds
def getAllData_5_feature():
    res = []
    data_1 = getData_5feature_tensor(1)
    data_2 = getData_5feature_tensor(2)
    data_5 = getData_5feature_tensor(5)
    data_18 = getData_5feature_tensor(18)
    data_19 = getData_5feature_tensor(19)
    data_21 = getData_5feature_tensor(21)
    data_22 = getData_5feature_tensor(22)
    data_30 = getData_5feature_tensor(30)
    data_52 = getData_5feature_tensor(52)
    res.append(data_1)
    res.append(data_2)
    res.append(data_5)
    #res.append(data_18)
    res.append(data_19)
    res.append(data_21)
    res.append(data_22)
    #res.append(data_30)
    res.append(data_52)

    return res

def getVal():
    res = []
    data_18 = getData_5feature_tensor(18)
    res.append(data_18)

    return res
def getTest_set():
    res = []
    data_30 = getData_5feature_tensor(30)

    res.append(data_30)
    return res

# Sliding windows used to split data to x and y
def sliding_windows(data, seq_length, n_future):
    x = []
    y = []

    for i in range(len(data)-seq_length):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length:i+seq_length+n_future]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

def Load_Data(data):
    seq_length = 4    # Choose params for sliding window, use 4 to predict 2 futures here
    n_future = 1
    x_de, y_de = sliding_windows(data, seq_length, n_future)

    train_size = int(len(y_de) * 0.7) # Split the datasets to train/val/test
    val_size = int(len(y_de) * 0.2)
    test_size = len(y_de) - train_size - val_size

    x_train = x_de[:train_size]
    y_train = y_de[:train_size]

    train_data = np.concatenate((x_train, y_train), axis=1)
    train_data_rs = np.reshape(train_data, (train_data.shape[0]*train_data.shape[1], train_data.shape[2]))
    scaler = StandardScaler()
    scaler.fit(train_data_rs)
    mean = scaler.mean_
    std = np.sqrt(scaler.var_)
    data_no = scaler.transform(data)

    x, y = sliding_windows(data_no, seq_length, n_future)
    dataX = Variable(torch.Tensor(np.array(x))) # Convert to tensor
    dataY = Variable(torch.Tensor(np.array(y)))

    trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y[0:train_size])))
    #trainY_time = trainY[:,:,2]
    trainY = trainY[:,:,0:2]
    
    valX = Variable(torch.Tensor(np.array(x[train_size:train_size+val_size])))
    valY = Variable(torch.Tensor(np.array(y[train_size:train_size+val_size])))
    #valY_time = valY[:,:,2]
    valY = valY[:,:,0:2]
    
    testX = Variable(torch.Tensor(np.array(x[train_size+val_size:len(x)])))
    testY = Variable(torch.Tensor(np.array(y[train_size+val_size:len(y)])))
    #testY_time = testY[:,:,2]
    testY = testY[:,:,0:2]




    return trainX, trainY, valX, valY, testX, testY, dataX, dataY, train_size, val_size, mean, std

def train_test_split(res):
    
    train_X = []
    train_Y = []
    val_X = []
    val_Y = []
    test_X = []
    test_Y = []
    data_X = []
    data_Y = []
      # Integrate all ponds data together to build one model
    for i in res:
        train_x, train_y, val_x, val_y, test_x, test_y, data_x, data_y,_,_, mean, std = Load_Data(i)
        train_X.extend(train_x)
        train_Y.extend(train_y)
        val_X.extend(val_x)
        val_Y.extend(val_y)
        test_X.extend(test_x)
        test_Y.extend(test_y)
        data_X.extend(data_x)
        data_Y.extend(data_y)
    train_X = torch.stack(train_X).to(device)
    train_Y = torch.stack(train_Y).to(device)
    val_X = torch.stack(val_X).to(device)
    val_Y = torch.stack(val_Y).to(device)
    test_X = torch.stack(test_X).to(device)
    test_Y = torch.stack(test_Y).to(device)
    data_X = torch.stack(data_X).to(device)
    data_Y = torch.stack(data_Y).to(device)

    return train_X, train_Y, val_X, val_Y, test_X, test_Y, data_X, data_Y, mean, std

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.best_model = model.state_dict().copy()
        self.val_loss_min = val_loss

class LSTM(nn.Module):
    def __init__(self, num_classes=2, input_size=3, hidden_size=512, num_layers=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
       
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.3 if num_layers > 1 else 0 
        )
        
       
        self.dropout1 = nn.Dropout(0.5) 
        self.dropout2 = nn.Dropout(0.4)  
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(128)
        
 
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(128, num_classes)
        
 
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化帮助防止梯度消失/爆炸"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x):

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        

        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        

        out = output[:, -1, :]
        
 
        out = self.bn1(out)
        out = self.dropout1(out)
        

        out = self.fc_1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        

        out = self.fc_2(out)
    
        out = out.view(out.size(0), 1, -1)
        
        return out
    


def convert_tensor(arr):
    last_sequence = Variable(torch.Tensor(np.array(arr)))
    last_sequence = last_sequence.unsqueeze(0)
    last_sequence = last_sequence.reshape(last_sequence.shape[1], last_sequence.shape[2])
    #last_sequence = scaler.transform(last_sequence)
    last_sequence = Variable(torch.Tensor(np.array(last_sequence)))
    last_sequence = last_sequence.unsqueeze(0)
    last_sequence = last_sequence.to(device)
    return last_sequence

def get_latest(pond_number):
    base_path = '/LH_Farm/pond_'
    pond = f"{base_path}{pond_number}"  #Get data from the firebase
    ref = db.reference(pond).order_by_key().limit_to_last(20)
    data = ref.get()
    df = pd.DataFrame(data)
    df = df.T
    
    df = df[df['type'] == 'a_buoy']
    df = df[df.index.str[:8] > '20250701']
    df_do_related = df[['do', 'temp', 'pressure']]
    df_init_do = df[['init_do']]

    #do_res = remove_outliers_from_df(df_do_related)
    do_mean = Convert_mean(df_do_related) # Get mean of DO
    do_mean = pd.concat([do_mean, df_init_do], axis=1)
    do_mean['do'] = do_mean.apply(lambda row: convert_to_mgl(100 * row['do'] / row['init_do'], row['temp'], row['pressure']), axis=1)

    do_mean['datetime'] = pd.to_datetime(do_mean.index, format='%Y%m%d_%H:%M:%S')

    do_mean['datetime'] = do_mean['datetime'].dt.tz_localize('UTC')
    do_mean['datetime'] = do_mean['datetime'].dt.tz_convert('America/Chicago')

    do_mean['hour'] = do_mean['datetime'].dt.hour
    do_mean['minute'] = do_mean['datetime'].dt.minute
    do_mean['hour_minute'] = do_mean['hour'] + do_mean['minute'] / 60.0

    do_mean['formatted_index'] = do_mean['datetime'].dt.strftime('%Y%m%d_%H:%M:%S')
    do_mean = do_mean.set_index('formatted_index', drop=True)
    do_mean = do_mean.drop('datetime', axis = 1)
    do_mean = do_mean.drop('hour', axis = 1)
    do_mean = do_mean.drop('minute', axis = 1)
    do_mean = do_mean.iloc[:-10]
    do_mean['datetime'] = pd.to_datetime(do_mean.index, format='%Y%m%d_%H:%M:%S')

    do_mean_index = do_mean.index


    do_mean = do_mean[['do', 'temp', 'hour_minute']]
    do_mean = do_mean.dropna()
    with open('latest_data.json', 'w') as f:
        json.dump(data, f)

    return do_mean



def Predict(pond_id, n_ahead, model):


    data = get_latest(pond_id)
    future_predicts = []
    _, _, _, _, _, _, dataX, dataY, train_size, val_size, mean, std = Load_Data(data)
    do = data.iloc[:, :]


    lastDay = do.tail(4) 
    last_hour_minute = lastDay['hour_minute'].iloc[-1]
    hold = lastDay
    lastDay = norimalize(lastDay, mean, std)
    

    last_date = data.index[-1]
    last_date = pd.to_datetime(data.index[-1], format='%Y%m%d_%H:%M:%S')
    
    predicted_datetimes = []
    cur_hour_minute = last_hour_minute


    for i in range(n_ahead):
 
        next_time_minutes = (cur_hour_minute * 60 + 10) % (24 * 60) 
        next_hour_minute = next_time_minutes / 60.0  
        

        last_date = last_date + timedelta(minutes=10)
        predicted_datetimes.append(last_date)
        
        cur_hour_minute = next_hour_minute


        last_sequence = convert_tensor(lastDay)
        

        model.eval()
        with torch.no_grad():
            future_predict = model(last_sequence)
        

        future_predict = future_predict[0, :, :] 
        future_predict = future_predict.detach().cpu().numpy()
        
       
        if future_predict.shape[1] == 2: 
            # 添加hour_minute特征
            hour_minute_column = np.full((future_predict.shape[0], 1), next_hour_minute)
            future_predict = np.concatenate((future_predict, hour_minute_column), axis=1)
        
 
        future_predict_de = denormalize(future_predict, mean, std)
        
   
        future_predict_de[:, 2] = next_hour_minute  
        
        future_predicts.append(future_predict_de)


        future_predict_de_norm = norimalize(future_predict_de, mean, std)
        
   
        lastDay = np.append(lastDay, future_predict_de_norm, axis=0)
        lastDay = lastDay[1:, :] 

    future_predicts = np.array(future_predicts)
    
    return dataX, do, future_predicts, train_size, val_size, mean, std


