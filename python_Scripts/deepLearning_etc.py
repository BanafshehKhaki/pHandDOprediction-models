# For 10 Sondes

# first input model
from keras.models import Sequential from keras.layers import Dense
from keras.layers import Dense
from keras.models import Sequential
from numpy import hstack
from numpy import array

#  Multi-headed MLP to forecast a dependent time series.
visible1 = Input(shape=(n_steps,))
dense1 = Dense(100, activation='relu')(visible1)
# second input model
visible2 = Input(shape=(n_steps,))
dense2 = Dense(100, activation='relu')(visible2)
# merge input models
merge = concatenate([dense1, dense2])
output = Dense(1)(merge)
model = Model(inputs=[visible1, visible2], outputs=output) model.compile(optimizer='adam', loss='mse')
# fit model
model.fit([X1, X2], y, epochs=2000, verbose=0)


# demonstrate prediction
x_input = array([[80, 85], [90, 95], [100, 105]])
x1 = x_input[:, 0].reshape((1, n_steps))
x2 = x_input[:, 1].reshape((1, n_steps))
yhat = model.predict([x1, x2], verbose=0)
print(yhat)


#
# Multi-output MLP Model
# We can then define one output layer for each of the three series that we wish to forecast, where each output submodel will forecast a single time step.
# define output 1
output1 = Dense(1)(dense)
# define output 2
output2 = Dense(1)(dense)
# define output 2
output3 = Dense(1)(dense)
model = Model(inputs=visible, outputs=[output1, output2, output3])
model.compile(optimizer='adam', loss='mse')

# separate output
y1 = y[:, 0].reshape((y.shape[0], 1))
y2 = y[:, 1].reshape((y.shape[0], 1))
y3 = y[:, 2].reshape((y.shape[0], 1))
# fit model
model.fit(X, [y1, y2, y3], epochs=2000, verbose=0)


# In practice, there is little difference to the MLP model in predicting a vector output that represents different output variables
# (as in the previous example) or a vector output that represents multiple time steps of one variable.
# Nevertheless, there are subtle and important differences in the way the training data is prepared.

# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_steps_in)) model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')


# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
    end_ix = i + n_steps_in
    out_end_ix = end_ix + n_steps_out
    # check if we are beyond the sequence if out_end_ix > len(sequence):
    break
    # gather input and output parts of the pattern
    seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix] X.append(seq_x)
    y.append(seq_y)
    return array(X), array(y)


# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_steps_in))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps_in))
yhat = model.predict(x_input, verbose=0)
print(yhat)


# Multivariate Multi-step MLP Models
# multivariate multi-step data preparation
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
    end_ix = i + n_steps_in
    # check if we are beyond the dataset if out_end_ix > len(sequences):
    out_end_ix = end_ix + n_steps_out-1
    break
    # gather input and output parts of the pattern
    seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1] X.append(seq_x)
    y.append(seq_y)
    return array(X), array(y)


# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])

# convert to [rows, columns] structure
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))  # horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))  # choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# convert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out) print(X.shape, y.shape)
# summarize the data
for i in range(len(X)):
    print(X[i], y[i])

(6, 3, 2)(6, 2)
[[10 15]
 [20 25]
 [30 35]][65 85]
[[20 25]
 [30 35]
 [40 45]][85 105]
[[30 35]
 [40 45]
 [50 55]][105 125]
[[40 45]
 [50 55]
 [60 65]][125 145]
[[50 55]
 [60 65]
 [70 75]][145 165]
[[60 65]
 [70 75]
 [80 85]][165 185]
# We can now develop an MLP model for multi-step predictions using a vector output. The complete example is listed below.
# multivariate multi-step mlp example
# split a multivariate sequence into samples

# flatten input
n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))
# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_input))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
x_input = array([[70, 75], [80, 85], [90, 95]])
x_input = x_input.reshape((1, n_input))
yhat = model.predict(x_input, verbose=0)
print(yhat)
# Running the example fits the model and predicts the next two time steps of the output
# sequence beyond the dataset. We would expect the next two steps to be [185, 205].

##################################################
# Multiple Parallel Input and Multi-step Output
##################################################

# Listing 7.83: Example output from preparing data for multi-step forecasting for a multivariate series.
(5, 3, 3)(5, 2, 3)
[[10 15 25]
 [20 25 45]
 [30 35 65]][[40 45 85][50 55 105]]
[[20 25 45]
 [30 35 65]
 [40 45 85]][[50 55 105][60 65 125]]
# We can now develop an MLP model to make multivariate multi-step forecasts.
# In addition to flattening the shape of the input data, as we have in prior examples,
# we must also flatten the three-dimensional structure of the output data.
# This is because the MLP model is only capable of taking vector inputs and outputs.
# multivariate multi-step mlp example
# split a multivariate sequence into samples


def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
    end_ix = i + n_steps_in
    out_end_ix = end_ix + n_steps_out
    # check if we are beyond the dataset if out_end_ix > len(sequences):
    break
    # gather input and output parts of the pattern
    seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
    X.append(seq_x)
    y.append(seq_y)
  return array(X), array(y)

in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure 
in_seq1 = in_seq1.reshape((len(in_seq1), 1)) 
in_seq2 = in_seq2.reshape((len(in_seq2), 1)) 
out_seq = out_seq.reshape((len(out_seq), 1)) 

# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# convert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# flatten input
n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))
# flatten output
n_output = y.shape[1] * y.shape[2]
y = y.reshape((y.shape[0], n_output))
# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_input)) model.add(Dense(n_output))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]]) x_input = x_input.reshape((1, n_input))
yhat = model.predict(x_input, verbose=0)
print(yhat)
# Listing 7.87: Example output from an MLP model for multi-step forecasting for a multivariate series.
