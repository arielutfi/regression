import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

sNamaFile = input("Masukan Nama File Dataset Training: ")

sInputMinScaler = input("Masukan Angka Min Scaler: ")
sInputMaxScaler = input("Masukan Angka Max Scaler: ")
iMinScaler = int(sInputMinScaler)
iMaxScaler = int(sInputMaxScaler)

sInputLookBack = input("Masukan Jumlah Look Back: ")
iLookBack = int(sInputLookBack)

sInputEpoch = input("Masukan Jumlah Epoch: ")
iEpoch = int(sInputEpoch)

dataframe = read_csv(sNamaFile, usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

# Normalisasi Data
scaler = MinMaxScaler(feature_range=(iMinScaler, iMaxScaler))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# #Pembagian dataset untuk train & test
# train_size = int(len(dataset) * 0.7)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# print(len(train), len(test))

train = dataset

#Pemodelan data train & test
look_back = iLookBack
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape'])
model.fit(trainX, trainY, epochs=iEpoch, batch_size=1, verbose=2)

# serialize model to JSON
model_json = model.to_json()
with open("reg_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("reg_model.h5")
print("Saved model to disk")