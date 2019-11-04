import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.models import model_from_json
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

numpy.random.seed(7)

sNamaFile = input("Masukan Nama File Dataset Testing: ")

sInputMinScaler = input("Masukan Angka Min Scaler: ")
sInputMaxScaler = input("Masukan Angka Max Scaler: ")
iMinScaler = int(sInputMinScaler)
iMaxScaler = int(sInputMaxScaler)

sInputLookBack = input("Masukan Jumlah Look Back: ")
iLookBack = int(sInputLookBack)

dataframe = read_csv(sNamaFile, usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

# Normalisasi Data
scaler = MinMaxScaler(feature_range=(iMinScaler, iMaxScaler))
dataset = scaler.fit_transform(dataset)

look_back = iLookBack
testX, testY = create_dataset(dataset, look_back)
trainX, trainY = create_dataset(dataset, look_back)

testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

# load json and create model
json_file = open('reg_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("reg_model.h5")
print("Loaded model from disk")

trainPredict = loaded_model.predict(trainX)
testPredict = loaded_model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))
mapeTrainScore = mean_absolute_percentage_error(trainY[0], trainPredict[:, 0])
print('Train Score: %.2f MAPE' % (mapeTrainScore))
mapeTestScore = mean_absolute_percentage_error(testY[0], testPredict[:, 0])
print('Test Score: %.2f MAPE' % (mapeTestScore))

trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[look_back:len(testPredict)+look_back, :] = testPredict

# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

plt.plot(scaler.inverse_transform(dataset), label='Expected')
# plt.plot(trainPredictPlot)
plt.plot(testPredictPlot, label='Predictions')
plt.plot(trainPredictPlot)
plt.plot(['mape'])
plt.xlabel('Day')
plt.ylabel('Testing Data Value')
plt.legend()
plt.show()

