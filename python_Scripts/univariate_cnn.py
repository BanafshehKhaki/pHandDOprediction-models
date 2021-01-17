# univariate cnn example
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import pandas as pd
import numpy as np
import sklearn.metrics as skm
import matplotlib.pyplot as plt
# split a univariate sequence into samples


# def split_sequence(sequence, n_steps):
#     X, y = list(), list()
#     for i in range(len(sequence)):
#         # find the end of this pattern
#         end_ix = i + n_steps
#         # check if we are beyond the sequence
#         if end_ix > len(sequence)-1:
#             break
#         # gather input and output parts of the pattern
#         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
#         X.append(seq_x)
#         y.append(seq_y)
#     return array(X), array(y)

def split_sequences(data, n_steps):
    data = data.values
    X, y = list(), list()

    for i in range(len(data)):
        end_ix = i + n_steps*6
        if end_ix > len(data):
            break

        Kx = np.empty((1, 12))
        for index in np.arange(i, i+(n_steps*6), step=6, dtype=int):
            eachhour = index + 6
            if eachhour > len(data) or i+(n_steps*6) > len(data):
                break

            a = data[index: eachhour, : -1]
            hourlymean_x = np.mean(a, axis=0)
            hourlymean_y = data[eachhour-1, -1]

            hourlymean_x = hourlymean_x.reshape((1, hourlymean_x.shape[0]))
            if index != i:
                Kx = np.append(Kx, hourlymean_x, axis=0)
            else:
                Kx = hourlymean_x

        X.append(Kx)
        y.append(hourlymean_y)
    print(np.array(X).shape)
    return np.array(X), np.array(y)


def temporal_horizon(df, pd_steps, target):
    pd_steps = pd_steps * 6
    target_values = df[[target]]
    target_values = target_values.drop(
        target_values.index[0: pd_steps], axis=0)
    target_values.index = np.arange(0, len(target_values[target]))

    df = df.drop(
        df.index[len(df.index)-pd_steps: len(df.index)], axis=0)
    df['Target_'+target] = target_values
    print('Target_'+target)
    return df


path = 'Sondes_data/train/train_data/'
file = 'leavon_wo_2019-07-01-2020-01-15.csv'
# define input sequence
dataset = pd.read_csv(path+file)
# dataset = dataset['dissolved_oxygen']

# raw_seq = dataset.values
# choose a number of time steps
n_steps = 3
df = temporal_horizon(dataset, 12, 'dissolved_oxygen')
df = df[['dissolved_oxygen', 'Target_dissolved_oxygen']]
# split into samples
X, y = split_sequences(df, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = X.shape[2]
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(Conv1D(64, 2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=1000, verbose=0)

# demonstrate prediction
testpath = 'Sondes_data/test/test_data/leavon_2019-07-01-2020-01-15.csv'
dataset = pd.read_csv(path+file)
n_steps = 3
df = temporal_horizon(dataset, 12, 'dissolved_oxygen')
df = df[['dissolved_oxygen', 'Target_dissolved_oxygen']]

# split into samples
X, y = split_sequences(df, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = X.shape[2]
X = X.reshape((X.shape[0], X.shape[1], n_features))
yhat = model.predict(X, verbose=0)
print(skm.r2_score(y, yhat))
plt.scatter(np.arange(len(y)),
            y, s=1)
plt.scatter(np.arange(len(yhat)),
            yhat, s=1)
plt.legend(['actual', 'predictions'],
           loc='upper right')
plt.show()
