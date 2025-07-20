import time

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.layers import (
    LSTM,
    BatchNormalization,
    Dense,
    Dropout,
    Input,
    LeakyReLU,
    RepeatVector,
    TimeDistributed,
)
from keras.models import Model, Sequential
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def lstm_autoencoder(input_tensor, n_nodes=64):
    lstm_model = LSTM(n_nodes, return_sequences=True)(input_tensor)

    dense1 = Dense(2 * n_nodes, activation="linear")(lstm_model)
    dense1 = LeakyReLU(alpha=0.1)(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.25)(dense1)

    dense2 = Dense(n_nodes, activation="linear")(dense1)
    dense2 = LeakyReLU(alpha=0.1)(dense2)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.10)(dense2)

    model_out = TimeDistributed(Dense(1, activation="linear"))(dense2)
    return Model(inputs=input_tensor, outputs=model_out)


# define Autoencoder model
# Input shape would be seq_size, 1 - 1 beacuse we have 1 feature.
# seq_size = trainX.shape[1]


def lstm_autoencoder2(input_tensor, n_nodes=64):
    model = Sequential()
    model.add(Input(shape=(input_tensor.shape[1], input_tensor.shape[2])))
    model.add(
        LSTM(
            n_nodes * 2,
            activation="relu",
            return_sequences=True,
        )
    )
    model.add(LSTM(n_nodes, activation="relu", return_sequences=False))
    model.add(RepeatVector(input_tensor.shape[1]))
    model.add(LSTM(n_nodes, activation="relu", return_sequences=False))
    model.add(LSTM(n_nodes * 2, activation="relu", return_sequences=False))
    model.add(TimeDistributed(Dense(input_tensor.shape[2])))
    return model


def to_sequences(x, y, seq_size=1):
    x_values = []
    y_values = []

    for i in range(len(x) - seq_size):
        x_values.append(x[i : (i + seq_size), :])
        y_values.append(y[i + seq_size, :])

    return tf.convert_to_tensor(x_values), tf.convert_to_tensor(y_values)


def main() -> None:
    dataframe = pd.read_csv("./data/international-airline-passengers.csv")
    data = dataframe.to_numpy()[:-2, :]
    # df = dataframe[["Date", "Close"]]
    times = list(data[:, 0].astype(str))
    #
    features = data[:, 1:].astype(float)
    times = np.array([float(x.split("-")[0] + "." + x.split("-")[1]) for x in times])
    # print(times)

    sns.lineplot(x=times, y=features[:, 1])
    plt.pause(0.001)

    print("Start time is: ", times.min())
    print("End time is: ", times.max())

    cutoff_time = 1959.07  # Mid 2017
    # Change train data from Mid 2017 to 2019.... seems to be a jump early 2017
    xtrain, xtest = features[times <= cutoff_time, :], features[times > cutoff_time, :]
    print(f"Shape of xtrain: {xtrain.shape}, xtest: {xtest.shape}")
    # normalize the dataset
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    scaler = scaler.fit(xtrain)

    xtrain = scaler.transform(xtrain)
    xtest = scaler.transform(xtest)
    # As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
    # In this example, the n_features is 2. We will make timesteps = 3.
    # With this, the resultant n_samples is 5 (as the input data has 9 rows).

    seq_size = 12  # Number of time steps to look back
    # Larger sequences (look further back) may improve forecasting.

    xs_train, ys_train = to_sequences(xtrain, xtrain, seq_size)
    xs_test, ys_test = to_sequences(xtest, xtest, seq_size)

    print(f"Shape of xs_train: {xs_train.shape}, ys_train: {ys_train.shape}")
    model = lstm_autoencoder2(Input(shape=xs_train.shape[1:]))

    model.compile(optimizer="adam", loss="mse")
    print(model.summary())

    # fit model
    history = model.fit(xs_train, ys_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.legend()
    plt.show()

    print(model.evaluate(xs_test, ys_test))

    ###########################
    # Anomaly is where reconstruction error is large.
    # We can define this value beyond which we call anomaly.
    # Let us look at MAE in training prediction

    train_predict = model.predict(xs_train)
    train_mae = np.mean(np.abs(train_predict - xs_train), axis=1)
    plt.hist(train_mae, bins=30)
    max_train_mae = 0.3  # or Define 90% value of max as threshold.
    plt.show()

    test_predict = model.predict(xs_test)
    test_mae = np.mean(np.abs(test_predict - xs_test), axis=1)
    plt.hist(test_mae, bins=30)
    plt.show()

    # Capture all details in a DataFrame for easy plotting
    anomaly_df = pd.DataFrame(xtest[seq_size:])
    anomaly_df["test_mae"] = test_mae
    anomaly_df["max_train_mae"] = max_train_mae
    anomaly_df["anomaly"] = anomaly_df["test_mae"] > anomaly_df["max_train_mae"]
    anomaly_df["Close"] = xtest[seq_size:]["Close"]

    # Plot test_mae vs max_train_mae
    sns.lineplot(x=anomaly_df["Date"], y=anomaly_df["test_mae"])
    sns.lineplot(x=anomaly_df["Date"], y=anomaly_df["max_train_mae"])
    plt.show()

    anomalies = anomaly_df.loc[anomaly_df["anomaly"]]

    # Plot anomalies
    sns.lineplot(x=anomaly_df["Date"], y=scaler.inverse_transform(anomaly_df["Close"]))
    sns.scatterplot(x=anomalies["Date"], y=scaler.inverse_transform(anomalies["Close"]), color="r")
    plt.show()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Program took {time.time() - t0:.3f} seconds.")
