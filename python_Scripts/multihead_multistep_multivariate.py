# multivariate multihead multistep
from numpy import array
from numpy import hstack
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.merge import concatenate
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
import time
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
import time
from sklearn.metrics import fbeta_score
from sklearn.metrics import r2_score


# def R2_measure(y_true, y_pred):
#     return r2_score(y_true, y_pred)


# def f2_measure(y_true, y_pred):
#     return fbeta_score(y_true, y_pred, labels=[1, 2],  beta=2, average='micro')


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

            a = data[index: eachhour, : -9]
            hourlymean_x = np.round(np.mean(a, axis=0), decimals=2)
            hourlymean_y = data[eachhour-1, -9:]

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
    # print('Target_'+target)
    return df


def algofind(model_name, neuron, input_dim, cat, n_steps_in, n_features, n_steps_out):
    if cat == 0:
        if model_name == 'multihead_MLP':
            visible1 = Input(shape=((n_steps_in * n_features),))
            dense1 = Dense(neuron, activation='relu')(visible1)
            # second input model
            visible2 = Input(shape=((n_steps_in * n_features),))
            dense2 = Dense(neuron, activation='relu')(visible2)

            # third input model
            visible3 = Input(shape=((n_steps_in * n_features),))
            dense3 = Dense(neuron, activation='relu')(visible3)
            # forth input model
            visible4 = Input(shape=((n_steps_in * n_features),))
            dense4 = Dense(neuron, activation='relu')(visible4)

            # fifth input model
            visible5 = Input(shape=((n_steps_in * n_features),))
            dense5 = Dense(neuron, activation='relu')(visible5)

            # merge input models
            merge = concatenate(
                [dense1, dense2, dense3, dense4, dense5])
            output = Dense(n_steps_out)(merge)
            model = Model(inputs=[visible1, visible2, visible3,
                                  visible4, visible5], outputs=output)
            model.compile(optimizer='adam', loss='mse')
    return model


def main():
    method = 'OrgData'

    # , 'DOcategory', 'pHcategory']  # ysi_blue_green_algae (has negative values for leavon... what does negative mean!?)
    targets = ['dissolved_oxygen', 'ph']

    models = ['multihead_MLP']
    path = 'Sondes_data/train_Summer/'
    files = [f for f in os.listdir(path) if f.endswith(
        ".csv") and f.startswith('leavon')]

    for model_name in models:
        for target in targets:
            if target.find('category') > 0:
                cat = 1
                directory = 'Results/bookThree/output_Cat_' + \
                    model_name+'/oversampling_cv_models/'
                data = {'CV': 'CV', 'target_names': 'target_names', 'method_names': 'method_names', 'temporalhorizons': 'temporalhorizons', 'window_nuggets': 'window_nuggets',
                        'file_names': 'file_names',  'F1_0': 'F1_0', 'F1_1': 'F1_1', 'P_0': 'P_0', 'P_1': 'P_1', 'R_0': 'R_0', 'R_1': 'R_1', 'acc0_1': 'acc0_1', 'F1_0_1': 'F1_0_1', 'F1_all': 'F1_all', 'fbeta': 'fbeta', 'configs': 'configs', 'scores': 'scores'}
            else:
                cat = 0
                directory = 'Results/bookThree/output_Reg_' + \
                    model_name+'/oversampling_cv_models/'
                data = {'target_names': 'target_names', 'method_names': 'method_names', 'window_nuggets': 'window_nuggets', 'temporalhorizons': 'temporalhorizons',  'CV': 'CV',
                        'file_names': 'file_names',  'mape': 'mape', 'me': 'me', 'mae': 'mae', 'mse': 'mse', 'rmse': 'rmse', 'R2': 'R2', 'configs': 'configs'}
            if not os.path.exists(directory):
                os.makedirs(directory)

            for file in files:

                result_filename = 'results_'+target + \
                    '_'+file + '_'+str(time.time())+'.csv'
                dfheader = pd.DataFrame(data=data, index=[0])
                dfheader.to_csv(directory+result_filename, index=False)
                PrH_index = 0
                for n_steps_in in [1, 3, 6, 12, 24, 36]:
                    print(n_steps_in)

                    dataset = pd.read_csv(path+file)
                    dataset = dataset[[
                        'year', 'month', 'day', 'hour', target]]

                    # dataset = dataset.dropna()
                    print(dataset.head())

                    dataset = temporal_horizon(
                        dataset, PrH_index, target)

                    train_X_grid, train_y_grid = split_sequences(
                        dataset, n_steps_in)

                    dataset_bgsusd = pd.read_csv(path+'bgsusd_all.csv')
                    dataset_osugi = pd.read_csv(path+'osugi.csv')
                    dataset_utlcp = pd.read_csv(path+'utlcp.csv')
                    dataset_leoc_1 = pd.read_csv(path+'leoc_1.csv')

                    dataset_bgsusd = temporal_horizon(
                        dataset_bgsusd[[target]], PrH_index, target)
                    dataset_osugi = temporal_horizon(
                        dataset_osugi[[target]], PrH_index, target)
                    dataset_utlcp = temporal_horizon(
                        dataset_utlcp[[target]], PrH_index, target)
                    dataset_leoc_1 = temporal_horizon(
                        dataset_leoc_1[[target]], PrH_index, target)

                    train_X_grid_bgsusd, train_y_grid_bgsusd = split_sequences(
                        dataset_bgsusd, n_steps_in)
                    train_X_grid_osugi, train_y_grid_osugi = split_sequences(
                        dataset_osugi, n_steps_in)
                    train_X_grid_utlcp, train_y_grid_utlcp = split_sequences(
                        dataset_utlcp, n_steps_in)
                    train_X_grid_leoc_1, train_y_grid_leoc_1 = split_sequences(
                        dataset_leoc_1, n_steps_in)

                    # print(train_X_grid[0:2])
                    # print("--")
                    input_dim = train_X_grid.shape
                    # print("shapes: ")
                    # print(input_dim)
                    # print(train_y_grid.shape)

                    # print('na:')
                    # inds = np.where(np.isnan(train_X_grid))
                    # print(inds)
                    # train_X_grid[inds] = 0
                    # inds = np.where(np.isnan(train_y_grid))
                    # train_y_grid[inds] = 0
                    # print(inds)
                    # print('--')
                    n_features = 1
                    X1 = train_X_grid[:, :, -1]
                    X2 = train_X_grid_bgsusd[:, :, -1]
                    X3 = train_X_grid_osugi[:, :, -1]
                    X4 = train_X_grid_utlcp[:, :, -1]
                    X5 = train_X_grid_leoc_1[:, :, -1]
                    y = train_y_grid

                    # print("-X-")
                    # print(X1.shape)
                    # print(np.array([X1, X2, X3, X4, X5]).shape)
                    # print("--")

                    n_steps_out = y.shape[1]
                    if cat:
                        y = to_categorical(y, 3)
                    # print(X1[0:2])
                    # print("--")

                    train_X_grid = train_X_grid.reshape(
                        train_X_grid.shape[0], train_X_grid.shape[1]*train_X_grid.shape[2])
                    # print(train_X_grid[0])
                    # dftime = pd.DataFrame({
                    #     'year': np.array(train_X_grid[:, -5]).astype(int), 'month': np.array(train_X_grid[:, -4]).astype(int),
                    #     'day': np.array(train_X_grid[:, -3]).astype(int), 'hour': np.array(train_X_grid[:, -2]).astype(int)})
                    # df_time = pd.to_datetime(
                    #     dftime, format='%Y%m%d %H')

                    # print(df_time.head())

                    start_time = time.time()

                    # if cat == 1:
                    #     metric = make_scorer(f2_measure)
                    # else:
                    #     metric = make_scorer(R2_measure)

                    # custom_cv = func.custom_cv_2folds(X1, 3)

                    # if cat == 1:
                    #     gs = RandomizedSearchCV(
                    #         estimator=model, param_distributions=func.param_grid['param_grid_'+model_name+str(cat)], n_iter=20, cv=custom_cv, scoring=metric,  verbose=0, random_state=42)
                    #     clf = gs.fit([X1, X2, X3, X4, X5], y, epochs=1000,
                    #                  model__class_weight={0: 1, 1: 50, 2: 100})
                    # else:
                    # gs = RandomizedSearchCV(
                    #     estimator=model, param_distributions=func.param_grid['param_grid_'+model_name+str(cat)], n_iter=1, cv=custom_cv, scoring=metric,  verbose=0, random_state=42)

                    i_cv = 1
                    neurons = [32, 64, 128]
                    epochs = [500, 1000, 2000]
                    custom_cv = func.custom_cv_2folds(train_X_grid, 3)
                    for train_index, test_index in custom_cv:
                        train_X = [X1[train_index], X2[train_index],
                                   X3[train_index], X4[train_index], X5[train_index]]
                        train_y = y[train_index]
                        test_X = [X1[test_index], X2[test_index],
                                  X3[test_index], X4[test_index], X5[test_index]]
                        test_y = y[test_index]

                        test_time = train_X_grid[test_index]
                        dftime = pd.DataFrame({
                            'year': np.array(test_time[:, -5]).astype(int), 'month': np.array(test_time[:, -4]).astype(int),
                            'day': np.array(test_time[:, -3]).astype(int), 'hour': np.array(test_time[:, -2]).astype(int),
                        })
                        df_time = pd.to_datetime(dftime, format='%Y%m%d %H')
                        # print("-CV test-")
                        # print(test_X[0:2])
                        # print(np.array(test_X).shape)
                        # print(test_y[0:2])
                        # print(np.array(test_y).shape)
                        # print("--")
                        # print("--")

                        for neuron in neurons:
                            for epoch in epochs:
                                model = algofind(
                                    model_name, neuron, input_dim, cat, n_steps_in, n_features, n_steps_out)
                                clf = model.fit(train_X, train_y,
                                                epochs=epoch, verbose=0)

                                configs = (neuron, epoch)
                                predictions = model.predict(test_X)

                                fpath = 'predictions_' + method+target+'_Window' +\
                                    str(n_steps_in) + '_TH' +\
                                    str(PrH_index)+'_CV' + \
                                    str(i_cv)+str(neuron)+str(epoch)+file

                                if cat == 1:
                                    test_y = np.argmax(test_y, axis=1)

                                cm0 = np.zeros((n_steps_out, 6))
                                for t in range(n_steps_out):
                                    cm0[t, :] = func.forecast_accuracy(
                                        predictions[:, t], test_y[:, t], cat)
                                print(cm0)

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
                                        ax[i, j].legend(['y', 'yhat'])
                                        j += 1
                                        if j > 1:
                                            i += 1
                                            j = 0

                                # plt.legend(['actual', 'predictions'],
                                #            loc='lower right')
                                plt.savefig(directory+fpath+'.jpg')
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

                                frames = [df_actual, df_predictions]
                                # concatenate dataframes
                                df = pd.concat(frames, axis=1)  # sort=False
                                df.to_csv(directory+fpath, index=False)

                                if cat == 1:
                                    data = {'target_names': target, 'method_names': method, 'window_nuggets': n_steps_in, 'temporalhorizons': PrH_index, 'CV': i_cv,
                                            'file_names': file, 'F1_0': cm0[0], 'F1_1': cm0[1], 'P_0': cm0[2], 'P_1': cm0[3], 'R_0': cm0[4], 'R_1': cm0[5], 'acc0_1': cm0[6], 'F1_0_1': cm0[7], 'F1_all': cm0[8], 'fbeta': [cm0[9]],  'configs': [configs]}
                                elif cat == 0:
                                    data = {'target_names': target, 'method_names': method, 'window_nuggets': n_steps_in, 'temporalhorizons': PrH_index, 'CV': i_cv,
                                            'file_names': file, 'mape': [cm0[:, 0]], 'me': [cm0[:, 1]], 'mae': [cm0[:, 2]], 'mse': [cm0[:, 3]], 'rmse': [cm0[:, 4]], 'R2': [cm0[:, 5]], 'configs': [configs]}

                                df = pd.DataFrame(data=data, index=[0])
                                df.to_csv(directory+result_filename,
                                          index=False, mode='a', header=False)

                                elapsed_time = time.time() - start_time
                                print(time.strftime("%H:%M:%S",
                                                    time.gmtime(elapsed_time)))
                        i_cv = i_cv+1


if __name__ == "__main__":
    main()
