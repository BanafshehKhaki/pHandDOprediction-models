# multivariate multihead multistep
from keras.layers import Flatten
from keras.layers import ConvLSTM2D
from keras.layers.merge import concatenate
from numpy import array
from numpy import hstack
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import os
import gc
import joblib
import functions as func
from sklearn.model_selection import RandomizedSearchCV
from keras.utils import to_categorical
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
import time
from sklearn.metrics import fbeta_score
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import Adam
from keras.constraints import maxnorm
from keras.layers import Dropout
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import pickle
import tensorflow as tf


def R2_measure(y_true, y_pred):
    return r2_score(y_true, y_pred)


def f2_measure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, labels=[1, 2],  beta=2, average='micro')


def split_sequences(data, n_steps, n_step_out):
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

            a = data[index: eachhour, : (-1*n_step_out)]
            hourlymean_x = np.round(np.mean(a, axis=0), decimals=2)
            hourlymean_y = data[eachhour-1, (-1*n_step_out):]

            hourlymean_x = hourlymean_x.reshape((1, hourlymean_x.shape[0]))
            if index != i:
                Kx = np.append(Kx, hourlymean_x, axis=0)
            else:
                Kx = hourlymean_x

        X.append(Kx)
        y.append(hourlymean_y)
    # print(np.array(X).shape)
    return np.array(X), np.array(y)


def temporal_horizon(df, pd_steps, target):
    for pd_steps in [1, 3, 6, 12, 24, 36, 48, 60, 72]:
        pd_steps = pd_steps * 6
        target_values = df[[target]]
        target_values = target_values.drop(
            target_values.index[0: pd_steps], axis=0)
        target_values.index = np.arange(0, len(target_values[target]))
        df['Target_'+target+'_t'+str(pd_steps)] = target_values

    df = df.drop(df.index[len(df.index)-(72*6): len(df.index)], axis=0)
    return df


def create_reg_LSTM_model(input_dim, n_steps_in, n_features, n_steps_out, neurons=1, learn_rate=0.01, dropout_rate=0.0, weight_constraint=0, activation='sigmoid'):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Reshape(target_shape=(
        n_steps_in*totalsets, n_features), input_shape=(n_steps_in*n_features*totalsets,)))
    model.add(tf.keras.layers.LSTM(neurons, activation=activation, return_sequences=True,
                                   kernel_constraint=tf.keras.constraints.MaxNorm(weight_constraint)))

    model.add(tf.keras.layers.Dropout(dropout_rate))
    # , return_sequences=True))
    model.add(tf.keras.layers.LSTM(neurons, activation=activation))
    # model.add(Dense(neurons, activation=activation))  # Adding new layer
    model.add(tf.keras.layers.Dense(n_steps_out))
    opt = tf.keras.optimizers.Adam(lr=learn_rate)
    model.compile(loss='mse', optimizer=opt)
    # model.save(save_weights_only=True, best_model_only=True)
    print('model: ' + str(model))
    return model


def create_reg_NN_model(input_dim, n_steps_in, n_features, n_steps_out, neurons=1, learn_rate=0.01, dropout_rate=0.0, weight_constraint=0, activation='sigmoid'):
    model = Sequential()
    model.add(Dense(neurons, activation=activation,
                    kernel_constraint=maxnorm(weight_constraint), input_shape=(n_steps_in*n_features*totalsets,)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(neurons, activation=activation))  # adding new layer
    model.add(Dense(n_steps_out))
    opt = Adam(lr=learn_rate)
    model.compile(loss='mse', optimizer=opt)
    print('model: ' + str(model))
    return model


def create_reg_endecodeLSTM_model(input_dim, n_steps_in, n_features, n_steps_out, neurons=1, learn_rate=0.01, activation='sigmoid'):
    model = Sequential()
    model.add(Reshape(target_shape=(
        n_steps_in*totalsets, n_features), input_shape=(n_steps_in*n_features*totalsets,)))

    model.add(LSTM(neurons, activation=activation,
                   input_shape=(n_steps_in*totalsets, n_features)))
    model.add(RepeatVector(1))
    model.add(LSTM(neurons, activation=activation, return_sequences=True))

    model.add(TimeDistributed(Dense(n_steps_out)))
    opt = Adam(lr=learn_rate)
    model.compile(loss='mse', optimizer=opt)
    return model


def create_reg_CNNenLSTM_model(input_dim, n_steps_in, n_features, n_steps_out, neurons=1, learn_rate=0.01, activation='sigmoid'):
    model = Sequential()
    model.add(Reshape(target_shape=(
        n_steps_in*totalsets, n_features), input_shape=(n_steps_in*n_features*totalsets,)))

    model.add(Conv1D(64, 1, activation=activation,
                     input_shape=(n_steps_in*totalsets, n_features)))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(RepeatVector(1))
    model.add(LSTM(neurons, activation=activation, return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation=activation)))
    model.add(TimeDistributed(Dense(n_steps_out)))
    opt = Adam(lr=learn_rate)
    model.compile(loss='mse', optimizer=opt)
    return model


def create_reg_ConvLSTM_model(input_dim, n_steps_in, n_features, n_steps_out, neurons=1, learn_rate=0.01, activation='sigmoid'):
    # reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features(channels)]
    model = Sequential()
    model.add(Reshape(target_shape=(
        n_steps_in, totalsets, n_features, 1), input_shape=(n_steps_in*n_features*totalsets,)))
    model.add(ConvLSTM2D(64, (1, 3), activation=activation,
                         input_shape=(n_steps_in, totalsets, n_features, 1)))
    model.add(Flatten())
    model.add(RepeatVector(1))
    model.add(LSTM(neurons, activation=activation, return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation=activation)))
    model.add(TimeDistributed(Dense(n_steps_out)))
    opt = Adam(lr=learn_rate)
    model.compile(loss='mse', optimizer=opt)
    return model


def algofind(model_name, input_dim, cat, n_features, n_steps_out, params, n_jobs):
    if cat == 0:
        if model_name == 'endecodeLSTM':
            activation, neurons, batch_size, epochs, learn_rate, n_steps_in = params
            model = KerasRegressor(build_fn=create_reg_endecodeLSTM_model, input_dim=input_dim, activation=activation,
                                   epochs=epochs, batch_size=batch_size,  neurons=neurons, n_steps_in=int(n_steps_in), learn_rate=learn_rate, n_features=int(n_features), n_steps_out=int(n_steps_out), verbose=0)
        elif model_name == 'CNNLSTM':
            activation, neurons, batch_size, epochs, learn_rate, n_steps_in = params
            model = KerasRegressor(build_fn=create_reg_CNNenLSTM_model, input_dim=input_dim, activation=activation,
                                   epochs=epochs, batch_size=batch_size,  neurons=neurons, n_steps_in=int(n_steps_in), learn_rate=learn_rate, n_features=int(n_features), n_steps_out=int(n_steps_out), verbose=0)
        elif model_name == 'ConvEnLSTM':
            # activation, neuron, bach size, epoch, learning rate, n_steps_in
            activation, neurons, batch_size, epochs, learn_rate, n_steps_in = params
            model = KerasRegressor(build_fn=create_reg_ConvLSTM_model, input_dim=input_dim, activation=activation,
                                   epochs=epochs, batch_size=batch_size,  neurons=neurons, n_steps_in=int(n_steps_in), learn_rate=learn_rate, n_features=int(n_features), n_steps_out=int(n_steps_out), verbose=0)

        elif model_name == 'NN':
            activation, neurons, batch_size, epochs, learn_rate, dropout_rate, weight_constraint, n_steps_in = params
            model = KerasRegressor(build_fn=create_reg_NN_model, epochs=epochs, batch_size=batch_size,
                                   input_dim=input_dim, n_steps_in=int(n_steps_in), n_features=int(n_features), n_steps_out=int(n_steps_out), neurons=neurons, learn_rate=learn_rate,
                                   dropout_rate=dropout_rate, weight_constraint=weight_constraint, activation=activation, verbose=0)
        elif model_name == 'LSTM':  # _oneSonde
            activation, neurons, batch_size, epochs, learn_rate, dropout_rate, weight_constraint, n_steps_in = params
            model = KerasRegressor(build_fn=create_reg_LSTM_model, epochs=epochs, batch_size=batch_size,
                                   input_dim=input_dim, n_steps_in=int(n_steps_in), n_features=int(n_features), n_steps_out=int(n_steps_out), neurons=neurons, learn_rate=learn_rate,
                                   dropout_rate=dropout_rate, weight_constraint=weight_constraint, activation=activation, verbose=0)

        elif model_name == 'DT':
            min_samples_split, max_features, criterion, max_depth,  min_samples_leaf,  n_steps = params
            model = MultiOutputRegressor(DecisionTreeRegressor(max_depth=max_depth,
                                                               criterion=criterion,
                                                               min_samples_split=min_samples_split,
                                                               min_samples_leaf=min_samples_leaf,
                                                               max_features=max_features))
        elif model_name == 'RF':
            max_features, n_estimators, max_depth,  min_samples_leaf, min_samples_split, bootstrap,  n_steps = params
            model = MultiOutputRegressor(RandomForestRegressor(max_depth=max_depth,
                                                               bootstrap=bootstrap,
                                                               n_estimators=n_estimators,
                                                               min_samples_split=min_samples_split,
                                                               min_samples_leaf=min_samples_leaf,
                                                               max_features=max_features))

        elif model_name == 'DT_onereg':
            # split, maxfeatures, criterion, minsampleleaf, maxdepth, lag
            min_samples_split, max_features, criterion, max_depth,  min_samples_leaf,  n_steps = params
            model = DecisionTreeRegressor(max_depth=max_depth,
                                          criterion=criterion,
                                          min_samples_split=min_samples_split,
                                          min_samples_leaf=min_samples_leaf,
                                          max_features=max_features, n_jobs=n_jobs)
        elif model_name == 'RF_onereg':
            # maxfeature, n_estimator, maxdepth, minsampleleaf, min_samplesplit, bootstrap, lag
            max_features, n_estimators, max_depth,  min_samples_leaf, min_samples_split, bootstrap,  n_steps = params
            model = RandomForestRegressor(max_depth=max_depth,
                                          bootstrap=bootstrap,
                                          n_estimators=n_estimators,
                                          min_samples_split=min_samples_split,
                                          min_samples_leaf=min_samples_leaf,
                                          max_features=max_features, n_jobs=n_jobs)

        elif model_name == 'SVC':
            epsilon, gamma, C, n_steps = params
            model = MultiOutputRegressor(
                SVR(epsilon=epsilon, gamma=gamma, C=C))

    return model


totalsets = 2


def main():

    # models = ['endecodeLSTM', 'CNNLSTM', 'ConvEnLSTM',
    #           'NN', 'SVC', 'RF_onereg', 'DT_onereg']

    models = ['LSTM']  # save the models later
    # 'DOcategory', 'pHcategory','ph', 'dissolved_oxygen',
    targets = ['dissolved_oxygen', 'ph']
    path = 'Sondes_data/train_Summer/'
    # files = [f for f in os.listdir(path) if f.endswith(
    #     ".csv") and f.startswith('leavon')]  # leavon
    files = ['osugi.csv', 'utlcp.csv',
             'leoc_1.csv', 'leavon.csv']
    n_job = -1
    PrH_index = 0

    for model_name in models:
        print(model_name)
        for target in targets:
            print(target)
            if target.find('category') > 0:
                cat = 1
                directory = 'Results/bookThree/2sondes/output_Cat_' + \
                    model_name+'/final_models/'
                data = {'target_names': 'target_names', 'method_names': 'method_names', 'window_nuggets': 'window_nuggets', 'temporalhorizons': 'temporalhorizons', 'CV': 'CV',
                        'file_names': 'file_names',  'F1_0': 'F1_0', 'F1_1': 'F1_1', 'P_0': 'P_0', 'P_1': 'P_1', 'R_0': 'R_0', 'R_1': 'R_1', 'acc0_1': 'acc0_1', 'F1_0_1': 'F1_0_1', 'F1_all': 'F1_all', 'fbeta': 'fbeta'}
            else:
                cat = 0
                directory = 'Results/bookThree/2sondes/output_Reg_' + \
                    model_name+'/final_models/'
                data = {'target_names': 'target_names', 'method_names': 'method_names', 'window_nuggets': 'window_nuggets', 'temporalhorizons': 'temporalhorizons', 'CV': 'CV',
                        'file_names': 'file_names',  'mape': 'mape', 'me': 'me', 'mae': 'mae', 'mse': 'mse', 'rmse': 'rmse',  'R2': 'R2'}

            if not os.path.exists(directory):
                os.makedirs(directory)
            print(directory)
            directoryresult = directory + 'Results/'
            if not os.path.exists(directoryresult):
                os.makedirs(directoryresult)

            # resultFileName = 'results_'+target+str(time.time())+'.csv'

            for file in files:

                method = 'OrgData'

                params = func.trained_param_grid[
                    'param_grid_'+model_name+str(cat)]
                n_steps_in = func.getlags_window(
                    model_name, params['param_'+target+'_'+str(PrH_index)], cat)
                print(n_steps_in)

                dataset = pd.read_csv(path+file)

                dataset = dataset[['Water_Temperature_at_Surface', 'ysi_chlorophyll',
                                   'dissolved_oxygen_saturation', 'dissolved_oxygen', 'ph',  'year', 'month', 'day', 'hour']]
                # print(dataset.head())

                dataset_bgsusd = pd.read_csv(path+'bgsusd_all.csv')

                dataset_bgsusd = dataset_bgsusd[['Water_Temperature_at_Surface', 'ysi_chlorophyll',
                                                 'dissolved_oxygen_saturation', 'dissolved_oxygen', 'ph', 'year', 'month', 'day', 'hour']]

                dataset = temporal_horizon(
                    dataset, PrH_index, target)

                dataset_bgsusd = temporal_horizon(
                    dataset_bgsusd, PrH_index, target)

                n_steps_out = 9
                train_X_grid, y = split_sequences(
                    dataset, n_steps_in, n_steps_out)
                print(train_X_grid.shape)

                n_features = train_X_grid.shape[2]
                print('n_fetures: ' + str(n_features))

                train_X_grid_bgsusd, train_y_grid_bgsusd = split_sequences(
                    dataset_bgsusd, n_steps_in, n_steps_out)

                train_X_grid = train_X_grid.reshape(
                    train_X_grid.shape[0], train_X_grid.shape[1]*train_X_grid.shape[2])

                train_X_grid_bgsusd = train_X_grid_bgsusd.reshape(
                    train_X_grid_bgsusd.shape[0], train_X_grid_bgsusd.shape[1]*train_X_grid_bgsusd.shape[2])

                XX = hstack((train_X_grid_bgsusd, train_X_grid))
                # XX = train_X_grid  # for final multivariate training model on LSTM
                print(XX.shape)
                # print(XX[0])
                input_dim = XX.shape

                start_time = time.time()

                model = algofind(
                    model_name, input_dim, cat, n_features, n_steps_out, params['param_'+target+'_'+str(PrH_index)], n_job)

                if model_name == 'RF' or model_name == 'DT':
                    pipeline = Pipeline(steps=[('model', model)])
                else:
                    pipeline = Pipeline(
                        steps=[('n', StandardScaler()), ('model', model)])

                # save the model to disk
                filename = model_name+'_model_' + \
                    target+'.joblib'

                if model_name == 'ConvEnLSTM' or model_name == 'endecodeLSTM' or model_name == 'CNNLSTM':
                    clf = pipeline.fit(XX, y.reshape(
                        y.shape[0], 1, n_steps_out))
                else:
                    clf = pipeline.fit(XX, y)

                # joblib.dump(clf, directory+filename)
                # pickle.dump(clf, open(directory+filename, 'wb'))

                # To load the model, open the file in reading and binary mode
                # load_lr_model =pickle.load(open(filename, 'rb'))

                elapsed_time = time.time() - start_time
                print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

                #################################
                # Testing final model on test data
                #################################
                start_time = time.time()
                testpath = 'Sondes_data/test_Summer/'
                # testfiles = ['lelorain.csv', 'utlcp.csv',
                #              'lementor_1.csv', 'lebiww.csv']
                # for testfile in testfiles:
                testfile = file
                result_filename = 'results_'+testfile+'_'+target + \
                    '_'+file+'_'+str(time.time())+'.csv'
                dfheader = pd.DataFrame(data=data, index=[0])
                dfheader.to_csv(directory+result_filename, index=False)

                dataset = pd.read_csv(testpath+testfile)
                dataset = dataset[['Water_Temperature_at_Surface', 'ysi_chlorophyll',
                                   'dissolved_oxygen_saturation', 'dissolved_oxygen', 'ph',  'year', 'month', 'day', 'hour']]
                # print(dataset.head())

                dataset_bgsusd = pd.read_csv(testpath+'bgsusd_all.csv')
                dataset_bgsusd = dataset_bgsusd[['Water_Temperature_at_Surface', 'ysi_chlorophyll',
                                                 'dissolved_oxygen_saturation', 'dissolved_oxygen', 'ph', 'year', 'month', 'day', 'hour']]

                dataset = temporal_horizon(
                    dataset, PrH_index, target)

                dataset_bgsusd = temporal_horizon(
                    dataset_bgsusd, PrH_index, target)

                test_X_grid, y = split_sequences(
                    dataset, n_steps_in, n_steps_out)

                n_features = test_X_grid.shape[2]
                test_X_grid_bgsusd, test_y_grid_bgsusd = split_sequences(
                    dataset_bgsusd, n_steps_in, n_steps_out)

                test_X_grid = test_X_grid.reshape(
                    test_X_grid.shape[0], test_X_grid.shape[1]*test_X_grid.shape[2])

                test_X_grid_bgsusd = test_X_grid_bgsusd.reshape(
                    test_X_grid_bgsusd.shape[0], test_X_grid_bgsusd.shape[1]*test_X_grid_bgsusd.shape[2])

                test_XX = hstack((test_X_grid_bgsusd, test_X_grid))
                # test_XX = test_X_grid

                i_cv = 1
                custom_cv = func.custom_cv_kfolds_testdataonly(
                    test_XX, 100)
                for test_index in custom_cv:
                    test_X = test_XX[test_index]
                    test_y = y[test_index]

                    test_time = test_XX[test_index]
                    # print(test_time[0])
                    dftime = pd.DataFrame({
                        'year': np.array(test_time[:, -4]).astype(int), 'month': np.array(test_time[:, -3]).astype(int),
                        'day': np.array(test_time[:, -2]).astype(int), 'hour': np.array(test_time[:, -1]).astype(int),
                    })
                    # print(dftime.head())
                    df_time = pd.to_datetime(dftime, format='%Y%m%d %H')

                    predictions = clf.predict(test_X)

                    # print(predictions.shape)
                    predictions = predictions.reshape(-1, n_steps_out)

                    fpath = 'predictions_' + method+target+'_Window' +\
                        str(n_steps_in) + '_CV' + str(i_cv)+testfile

                    if i_cv % 10 == 0:
                        fig, ax = plt.subplots(
                            nrows=5, ncols=2,  figsize=(50, 50))
                        i = j = 0
                        k = 0
                        columns = ['t+1', 't+3', 't+6', 't+12',
                                   't+24', 't+36', 't+48', 't+60', 't+72']
                        for col in columns:
                            if k < len(columns):
                                ax[i, j].scatter(
                                    df_time.values, test_y[:, k])
                                ax[i, j].scatter(
                                    df_time.values, predictions[:, k])
                                k = k+1
                                ax[i, j].set_title(col)
                                ax[i, j].legend(['actual', 'prediction'])
                                j += 1
                                if j > 1:
                                    i += 1
                                    j = 0

                        plt.savefig(directoryresult+fpath+'.png')
                        plt.close()

                        # print(test_y.shape)
                        # print(predictions.shape)
                        columns = ['a+1', 'a+3', 'a+6', 'a+12',
                                   'a+24', 'a+36', 'a+48', 'a+60', 'a+72']
                        df_actual = pd.DataFrame(
                            data=test_y, columns=columns)
                        columns = ['p+1', 'p+3', 'p+6', 'p+12',
                                   'p+24', 'p+36', 'p+48', 'p+60', 'p+72']
                        df_predictions = pd.DataFrame(
                            data=predictions, columns=columns)

                        frames = [df_time, df_actual, df_predictions]
                        # concatenate dataframes
                        df = pd.concat(frames, axis=1)  # , sort=False
                        df.to_csv(directoryresult+fpath, index=False)

                    cm0 = np.zeros((n_steps_out, 6))
                    for t in range(n_steps_out):
                        cm0[t, :] = func.forecast_accuracy(
                            predictions[:, t], test_y[:, t], cat)

                    if cat == 1:
                        data = {'target_names': target, 'method_names': method, 'window_nuggets': n_steps_in, 'temporalhorizons': PrH_index, 'CV': i_cv,
                                'file_names': testfile,  'F1_0': cm0[0], 'F1_1': cm0[1], 'P_0': cm0[2], 'P_1': cm0[3], 'R_0': cm0[4], 'R_1': cm0[5], 'acc0_1': cm0[6], 'F1_0_1': cm0[7], 'F1_all': cm0[8], 'fbeta': [cm0[9]]}
                    elif cat == 0:
                        data = {'target_names': target, 'method_names': method, 'window_nuggets': n_steps_in, 'temporalhorizons': PrH_index, 'CV': i_cv,
                                'file_names': testfile,  'mape': [cm0[:, 0]], 'me': [cm0[:, 1]], 'mae': [cm0[:, 2]], 'mse': [cm0[:, 3]], 'rmse': [cm0[:, 4]], 'R2': [cm0[:, 5]]}

                    df = pd.DataFrame(data=data, index=[0])
                    df.to_csv(directory+result_filename,
                              index=False, mode='a', header=False)

                    elapsed_time = time.time() - start_time
                    # print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                    i_cv = i_cv+1


if __name__ == "__main__":
    main()
