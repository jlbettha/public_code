import numpy as np
import pandas as pd
import seaborn as sns
import time
from keras.layers import (
    LSTM,
    Input,
    Dropout,
    Dense,
    RepeatVector,
    TimeDistributed,
    Concatenate,
    BatchNormalization,
    LeakyReLU,
)
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Model, Sequential


def LSTM_autoencoder(input_tensor, n_nodes=64, return_sequences=False):
    LSTM_model = LSTM(n_nodes, input_shape=input_tensor.shape, return_sequences=True)(
        input_tensor
    )

    dense1 = Dense(2 * n_nodes, activation="linear")(LSTM_model)
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


def LSTM_autoencoder2(X):
    model = Sequential()
    model.add(
        LSTM(
            128,
            activation="relu",
            input_shape=(X.shape[1], X.shape[2]),
            return_sequences=True,
        )
    )
    model.add(LSTM(64, activation="relu", return_sequences=False))
    model.add(RepeatVector(X.shape[1]))
    model.add(LSTM(64, activation="relu", return_sequences=False))
    model.add(LSTM(128, activation="relu", return_sequences=False))
    model.add(TimeDistributed(Dense(X.shape[2])))
    return model


def to_sequences(x, y, seq_size=1):
    x_values = []
    y_values = []

    for i in range(len(x) - seq_size):
        x_values.append(x[i : (i + seq_size), :])
        y_values.append(y[i + seq_size, :])

    return np.array(x_values), np.array(y_values)


def main() -> None:
    dataframe = pd.read_csv(
        "./temporal_sequential/data/international-airline-passengers.csv"
    )
    data = dataframe.to_numpy()[:-2, :]
    # df = dataframe[["Date", "Close"]]
    times = list(data[:, 0].astype(str))
    print(times)
    features = data[:, 1:].astype(float)
    times = np.array([float(x.split("-")[0] + "." + x.split("-")[1]) for x in times])

    sns.lineplot(x=times, y=features[:, 1])
    plt.pause(0.001)

    print("Start time is: ", times.min())
    print("End time is: ", times.max())

    # Change train data from Mid 2017 to 2019.... seems to be a jump early 2017
    Xtrain, Xtest = features[times <= 1959.07, :], features[times > 1959.07, :]

    # normalize the dataset
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    scaler = scaler.fit(Xtrain)

    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    # As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
    # In this example, the n_features is 2. We will make timesteps = 3.
    # With this, the resultant n_samples is 5 (as the input data has 9 rows).

    seq_size = 12  # Number of time steps to look back
    # Larger sequences (look further back) may improve forecasting.

    Xs_train, ys_train = to_sequences(Xtrain, Xtrain, seq_size)
    Xs_test, ys_test = to_sequences(Xtest, Xtest, seq_size)

    model = LSTM_autoencoder2(Xs_train)

    model.compile(optimizer="adam", loss="mse")
    print(model.summary())

    # fit model
    history = model.fit(
        Xs_train, ys_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1
    )

    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.legend()
    plt.show()

    print(model.evaluate(Xs_test, ys_test))

    ###########################
    # Anomaly is where reconstruction error is large.
    # We can define this value beyond which we call anomaly.
    # Let us look at MAE in training prediction

    trainPredict = model.predict(Xs_train)
    trainMAE = np.mean(np.abs(trainPredict - Xs_train), axis=1)
    plt.hist(trainMAE, bins=30)
    max_trainMAE = 0.3  # or Define 90% value of max as threshold.
    plt.show()

    testPredict = model.predict(Xs_test)
    testMAE = np.mean(np.abs(testPredict - Xs_test), axis=1)
    plt.hist(testMAE, bins=30)
    plt.show()

    # Capture all details in a DataFrame for easy plotting
    anomaly_df = pd.DataFrame(Xtest[seq_size:])
    anomaly_df["testMAE"] = testMAE
    anomaly_df["max_trainMAE"] = max_trainMAE
    anomaly_df["anomaly"] = anomaly_df["testMAE"] > anomaly_df["max_trainMAE"]
    anomaly_df["Close"] = Xtest[seq_size:]["Close"]

    # Plot testMAE vs max_trainMAE
    sns.lineplot(x=anomaly_df["Date"], y=anomaly_df["testMAE"])
    sns.lineplot(x=anomaly_df["Date"], y=anomaly_df["max_trainMAE"])
    plt.show()

    anomalies = anomaly_df.loc[anomaly_df["anomaly"] == True]

    # Plot anomalies
    sns.lineplot(x=anomaly_df["Date"], y=scaler.inverse_transform(anomaly_df["Close"]))
    sns.scatterplot(
        x=anomalies["Date"], y=scaler.inverse_transform(anomalies["Close"]), color="r"
    )
    plt.show()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Program took {time.time()-t0:.3f} seconds.")
