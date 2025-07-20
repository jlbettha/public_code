"""
Created on Mon Aug  7 17:07:57 2017

@author: jlbetthauser
"""

# LSTM for international airline passengers problem with regression framing
import math

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
    datax, datay = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i : (i + look_back), 0]
        datax.append(a)
        datay.append(dataset[i + look_back, 0])
    return np.array(datax), np.array(datay)


def main() -> None:

    # load the dataset
    dataframe = read_csv(
        "./data/international-airline-passengers.csv",
        usecols=[2],
        engine="python",
        skipfooter=3,
    )
    dataset = dataframe.to_numpy()
    dataset = dataset.astype(np.float32)

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.6)
    train, test = dataset[0:train_size, :], dataset[train_size : len(dataset), :]

    # reshape into X=t and Y=t+1
    look_back = 1
    trainx, trainy = create_dataset(train, look_back)
    testx, testy = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainx = np.reshape(trainx, (trainx.shape[0], 1, trainx.shape[1]))
    testx = np.reshape(testx, (testx.shape[0], 1, testx.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(trainx, trainy, epochs=30, batch_size=1, verbose=2)

    # make predictions
    train_predict = model.predict(trainx)
    test_predict = model.predict(testx)

    # invert predictions
    train_predict = scaler.inverse_transform(train_predict)
    trainy = scaler.inverse_transform([trainy])
    test_predict = scaler.inverse_transform(test_predict)
    testy = scaler.inverse_transform([testy])
    # ==============================================================================
    # trainY = numpy.reshape(trainY, (1,trainX.shape[0]))
    # testY = numpy.reshape(testY, (1,testX.shape[0]))
    # ==============================================================================

    # calculate root mean squared error
    train_score = math.sqrt(mean_squared_error(trainy[0], train_predict[:, 0]))
    print(f"Train Score: {train_score:.2f} RMSE")
    test_score = math.sqrt(mean_squared_error(testy[0], test_predict[:, 0]))
    print(f"Test Score: {test_score:.2f} RMSE")

    # shift train predictions for plotting
    train_predict_plot = np.empty_like(dataset)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back - 1 : len(train_predict) + look_back - 1, :] = (
        train_predict
    )

    # shift test predictions for plotting
    test_predict_plot = np.empty_like(dataset)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[
        len(train_predict) + (look_back * 2) + 1 - 1 : len(dataset) - 1 - 1, :
    ] = test_predict

    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(train_predict_plot)
    plt.plot(test_predict_plot)
    plt.show()


if __name__ == "__main__":
    main()
