import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import os
import gc
import joblib
from sklearn.metrics import r2_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import sklearn.metrics as skm
from sklearn.metrics import confusion_matrix
import time
import functions as func
import datetime
import univariatefunctions as ufunc
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed


def main():
    method = 'OrgData'

    # 'DOcategory', 'pHcategory',ysi_blue_green_algae (has negative values for leavon... what does negative mean!?)
    # 'ysi_blue_green_algae']  # , 'dissolved_oxygen', 'ph']
    targets = ['ph']
# 'ARIMA', 'SARIMA', 'ETS', 'AR', 'MA'
    models = ['SARIMA']
    path = 'Sondes_data/train_Summer/'
    files = [f for f in os.listdir(path) if f.endswith(
        ".csv") and f.startswith('leavon')]  # leavon bgsusd_all

    for model_name in models:
        for target in targets:
            if target.find('category') > 0:
                cat = 1
                directory = 'Results/bookThree/output_Cat_' + \
                    model_name+'/oversampling_cv_models/'
                data = {'CV': 'CV', 'target_names': 'target_names', 'method_names': 'method_names', 'temporalhorizons': 'temporalhorizons', 'window_nuggets': 'window_nuggets', 'config': 'config',
                        'file_names': 'file_names',  'F1_0': 'F1_0', 'F1_1': 'F1_1', 'P_0': 'P_0', 'P_1': 'P_1', 'R_0': 'R_0', 'R_1': 'R_1', 'acc0_1': 'acc0_1', 'F1_0_1': 'F1_0_1', 'F1_all': 'F1_all', 'fbeta': 'fbeta'}
            else:
                cat = 0
                directory = 'Results/bookThree/output_Reg_' + \
                    model_name+'/oversampling_cv_models/'
                data = {'CV': 'CV', 'target_names': 'target_names', 'method_names': 'method_names', 'temporalhorizons': 'temporalhorizons', 'window_nuggets': 'window_nuggets', 'config': 'config',
                        'file_names': 'file_names',  'mape': 'mape', 'me': 'me', 'mae': 'mae', 'mpe': 'mpe', 'rmse': 'rmse', 'R2': 'R2'}
            if not os.path.exists(directory):
                os.makedirs(directory)

            for file in files:
                print(file)
                result_filename = 'results_'+target + \
                    '_'+file + '_'+str(time.time())+'.csv'
                dfheader = pd.DataFrame(data=data, index=[0])
                dfheader.to_csv(directory+result_filename, index=False)
                n_steps = 1

                for PrH_index in [1, 3, 6, 12, 24, 36]:

                    dataset = pd.read_csv(path+file)

                    # Only the Target
                    dataset = dataset[[
                        'year', 'month', 'day', 'hour', target]]

                    print('Window: '+str(n_steps) + ' TH: ' +
                          str(PrH_index)+' '+method+' '+target)

                    i = 1

                    if model_name == 'MA':
                        train_X_grid, train_y_grid, input_dim, features = func.preparedata(
                            dataset, PrH_index, n_steps, target, cat)

                        start_time = time.time()
                        # For Train files:
                        custom_cv = func.custom_cv_2folds(train_X_grid, 3)
                        for train_index, test_index in custom_cv:
                            train_X = train_X_grid[train_index]
                            train_y = train_y_grid[train_index]
                            train_X_uni = train_X[:, -1]

                            test_X = train_X_grid[test_index]
                            # actual future values
                            test_X_uni = test_X[:, -1]
                            test_y = train_y_grid[test_index]

                            predictions = ufunc.movingAverage(
                                train_X_uni, train_y, test_X_uni, test_y)

                            df_time = pd.DataFrame({
                                'year': np.array(test_X[:, 0]).astype(int), 'month': np.array(test_X[:, 1]).astype(int),
                                'day': np.array(test_X[:, 2]).astype(int), 'hour': np.array(test_X[:, 3]).astype(int),
                            })

                            timeline = pd.to_datetime(
                                df_time, format='%Y%m%d %H')

                            if cat == 1:
                                predictions = np.array(predictions).astype(int)
                                test_y = np.array(test_y).astype(int)

                            # test_y = test_y.reshape(len(test_y),)
                            # predictions = predictions.reshape(
                            #     len(predictions),)

                            cm0 = func.forecast_accuracy(
                                predictions, test_y, cat)

                            filename = file + '_' + \
                                target+'_TH' + \
                                str(PrH_index)+'_lag' + \
                                str(n_steps)+'_'+str(i)

                            plt.scatter(timeline.values,
                                        test_y, s=1)
                            plt.scatter(timeline.values,
                                        predictions, s=1)
                            plt.legend(['actual', 'predictions'],
                                       loc='upper right')
                            plt.xticks(rotation=45)

                            directorydeeper = directory+'more/'
                            if not os.path.exists(directorydeeper):
                                os.makedirs(directorydeeper)
                            plt.savefig(directorydeeper+filename+'.jpg')

                            plt.close()
                            data = {'time': timeline,
                                    'Actual': test_y,
                                    'Predictions': predictions}
                            df = pd.DataFrame(data=data)

                            df.to_csv(directorydeeper+filename +
                                      '.csv', index=False)

                            if cat == 1:
                                data = {'CV': i, 'target_names': target, 'method_names': method, 'temporalhorizons': PrH_index, 'window_nuggets': 1,
                                        'file_names': filename,  'F1_0': cm0[0], 'F1_1': cm0[1], 'P_0': cm0[2], 'P_1': cm0[3], 'R_0': cm0[4], 'R_1': cm0[5], 'acc0_1': cm0[6], 'F1_0_1': cm0[7], 'F1_all': cm0[8], 'fbeta': [cm0[9]]}
                            elif cat == 0:
                                data = {'CV': i, 'target_names': target, 'method_names': method, 'temporalhorizons': PrH_index, 'window_nuggets': 1,
                                        'file_names': filename,  'mape': cm0[0], 'me': cm0[1], 'mae': cm0[2], 'mpe': cm0[3], 'rmse': cm0[4], 'R2': cm0[5]}

                            df = pd.DataFrame(data=data, index=[0])
                            df.to_csv(directory+result_filename,
                                      index=False, mode='a', header=False)
                            i = i + 1

                            elapsed_time = time.time() - start_time
                            print(time.strftime("%H:%M:%S",
                                                time.gmtime(elapsed_time)))

                    if model_name == 'ARIMA' or model_name == 'AR' or model_name == 'ETS' or model_name == 'SARIMA' or model_name == 'BL':
                        start_time = time.time()
                        train_X_grid = dataset.values
                        custom_cv = ufunc.custom_cv_2folds(
                            train_X_grid, 1, PrH_index)

                        ######################
                        # Cross Validation sets
                        ######################
                        i = 1
                        for train_index, test_index in custom_cv:
                            train_X = train_X_grid[train_index]
                            train_X_uni = train_X[:, -1]

                            test_X = train_X_grid[test_index]
                            # actual future values
                            test_X_uni = test_X[:, -1]

                            df_time = pd.DataFrame({
                                'year': np.array(test_X[:, 0]).astype(int), 'month': np.array(test_X[:, 1]).astype(int),
                                'day': np.array(test_X[:, 2]).astype(int), 'hour': np.array(test_X[:, 3]).astype(int),
                            })

                            timeline = pd.to_datetime(
                                df_time, format='%Y%m%d %H')

                            if model_name == 'BL':

                                # train_X_uni,test_X_uni
                                # make them into dataFrame so below can be done

                                test_X_uni = pd.DataFrame(test_X_uni)
                                target_values = test_X_uni.drop(
                                    test_X_uni.index[0: 1], axis=0)
                                target_values.index = np.arange(
                                    0, len(target_values))

                                # test_X_uni = pd.DataFrame(test_X_uni)

                                predictions = test_X_uni.drop(
                                    test_X_uni.index[len(test_X_uni)-1: len(test_X_uni)], axis=0)
                                test_X_uni = target_values

                                timeline = timeline.drop(
                                    timeline.index[len(timeline)-1: len(timeline)], axis=0)

                                cm0 = func.forecast_accuracy(
                                    predictions, test_X_uni, cat)

                                filename = file + '_' + \
                                    target+'_TH' + \
                                    str(PrH_index)+'_lag' + \
                                    str(n_steps)+'_'+str(i)

                                plt.scatter(timeline.values,
                                            test_X_uni, s=1)
                                plt.scatter(timeline.values,
                                            predictions, s=1)
                                plt.legend(['actual', 'predictions'],
                                           loc='upper right')
                                plt.xticks(rotation=45)

                                directorydeeper = directory+'more/'
                                if not os.path.exists(directorydeeper):
                                    os.makedirs(directorydeeper)
                                plt.savefig(directorydeeper+filename+'.jpg')

                                plt.close()

                                print(predictions.head())
                                print(test_X_uni.head())
                                print(timeline.head())

                                # data = {'time': timeline,
                                #         'Actual': test_X_uni,
                                #         'Predictions': predictions}
                                frames = [timeline, test_X_uni, predictions]
                                df = pd.concat(frames, axis=1)
                                df.to_csv(directorydeeper+filename +
                                          '.csv', index=False, header=['time', 'Actual', 'Predictions'])

                                if cat == 1:
                                    data = {'CV': i, 'target_names': target, 'method_names': method, 'temporalhorizons': PrH_index, 'window_nuggets': 1,
                                            'file_names': filename,  'F1_0': cm0[0], 'F1_1': cm0[1], 'P_0': cm0[2], 'P_1': cm0[3], 'R_0': cm0[4], 'R_1': cm0[5], 'acc0_1': cm0[6], 'F1_0_1': cm0[7], 'F1_all': cm0[8], 'fbeta': [cm0[9]]}
                                elif cat == 0:
                                    data = {'CV': i, 'target_names': target, 'method_names': method, 'temporalhorizons': PrH_index, 'window_nuggets': 1,
                                            'file_names': filename,  'mape': cm0[0], 'me': cm0[1], 'mae': cm0[2], 'mpe': cm0[3], 'rmse': cm0[4], 'R2': cm0[5]}

                                df = pd.DataFrame(data=data, index=[0])
                                df.to_csv(directory+result_filename,
                                          index=False, mode='a', header=False)

                            if model_name == 'AR':
                                predictions = ufunc.AutoRegression(
                                    train_X_uni, test_X_uni)
                                if cat == 1:
                                    predictions = np.array(
                                        predictions).astype(int)
                                    test_X_uni = np.array(
                                        test_X_uni).astype(int)

                                cm0 = func.forecast_accuracy(
                                    predictions, test_X_uni, cat)

                                filename = file + '_' + \
                                    target+'_TH' + \
                                    str(PrH_index)+'_lag' + \
                                    str(n_steps)+'_'+str(i)

                                plt.scatter(timeline.values,
                                            test_X_uni, s=1)
                                plt.scatter(timeline.values,
                                            predictions, s=1)
                                plt.legend(['actual', 'predictions'],
                                           loc='upper right')
                                plt.xticks(rotation=45)

                                directorydeeper = directory+'more/'
                                if not os.path.exists(directorydeeper):
                                    os.makedirs(directorydeeper)
                                plt.savefig(directorydeeper+filename+'.jpg')

                                plt.close()
                                data = {'time': timeline,
                                        'Actual': test_X_uni,
                                        'Predictions': predictions}
                                df = pd.DataFrame(data=data)

                                df.to_csv(directorydeeper+filename +
                                          '.csv', index=False)

                                if cat == 1:
                                    data = {'CV': i, 'target_names': target, 'method_names': method, 'temporalhorizons': PrH_index, 'window_nuggets': 1,
                                            'file_names': filename,  'F1_0': cm0[0], 'F1_1': cm0[1], 'P_0': cm0[2], 'P_1': cm0[3], 'R_0': cm0[4], 'R_1': cm0[5], 'acc0_1': cm0[6], 'F1_0_1': cm0[7], 'F1_all': cm0[8], 'fbeta': [cm0[9]]}
                                elif cat == 0:
                                    data = {'CV': i, 'target_names': target, 'method_names': method, 'temporalhorizons': PrH_index, 'window_nuggets': 1,
                                            'file_names': filename,  'mape': cm0[0], 'me': cm0[1], 'mae': cm0[2], 'mpe': cm0[3], 'rmse': cm0[4], 'R2': cm0[5]}

                                df = pd.DataFrame(data=data, index=[0])
                                df.to_csv(directory+result_filename,
                                          index=False, mode='a', header=False)

                            cfg_list = list()
                            if model_name == 'ETS':
                                cfg_list = ufunc.exp_smoothing_configs()
                                scores = [ufunc.score_model('ETS', train_X_uni, test_X_uni, cfg, cat, directory, file,
                                                            target, PrH_index, n_steps, i, result_filename, timeline) for cfg in cfg_list]

                            if model_name == 'ARIMA':
                                cfg_list = ufunc.ARIMA_configs()
                                scores = [ufunc.score_model('ARIMA', train_X_uni, test_X_uni, cfg, cat, directory,
                                                            file, target, PrH_index, n_steps, i, result_filename, timeline) for cfg in cfg_list]

                            if model_name == 'SARIMA':
                                cfg_list = ufunc.sarima_configs()

                                scores = [ufunc.score_model('SARIMA', train_X_uni, test_X_uni, cfg, cat, directory,
                                                            file, target, PrH_index, n_steps, i, result_filename, timeline) for cfg in cfg_list]

                            i = i + 1
                            elapsed_time = time.time() - start_time
                            print(time.strftime("%H:%M:%S",
                                                time.gmtime(elapsed_time)))


if __name__ == "__main__":
    main()
