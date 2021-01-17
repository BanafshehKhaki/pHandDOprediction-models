


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


def main():
    methods = ['OrgData']
    # 'dissolved_oxygen', 'ph', 'DOcategory', 'pHcategory']
    targets = ['dissolved_oxygen', 'ph']
    model_name = 'baseline'
    path = 'Sondes_data/test_Summer/'  # test_Summer train_Summer  #1
    files = [f for f in os.listdir(path) if f.endswith(".csv")]

    for method in methods:
        for target in targets:
            if target.find('category') > 0:
                cat = 1
                directory = 'Results/bookThree/output_Cat_' + \
                    model_name+'/final_models/Results/'  # final_models/Results  oversampling_cv_models/ #2
                data = {'CV': 'CV', 'target_names': 'target_names', 'method_names': 'method_names', 'temporalhorizons': 'temporalhorizons', 'window_nuggets': 'window_nuggets',
                        'file_names': 'file_names',  'F1_0': 'F1_0', 'F1_1': 'F1_1', 'P_0': 'P_0', 'P_1': 'P_1', 'R_0': 'R_0', 'R_1': 'R_1', 'acc0_1': 'acc0_1', 'F1_0_1': 'F1_0_1', 'F1_all': 'F1_all', 'fbeta': 'fbeta'}
            else:
                cat = 0
                directory = 'Results/bookThree/output_Reg_' + \
                    model_name+'/final_models/Results/'  # final_models/Results  oversampling_cv_models  #3
                data = {'CV': 'CV', 'target_names': 'target_names', 'method_names': 'method_names', 'temporalhorizons': 'temporalhorizons', 'window_nuggets': 'window_nuggets',
                        'file_names': 'file_names',  'mape': 'mape', 'me': 'me', 'mae': 'mae', 'mpe': 'mpe', 'rmse': 'rmse', 'R2': 'R2'}
            if not os.path.exists(directory):
                os.makedirs(directory)
            for file in files:
                print(file)
                result_filename = 'results_'+target+'_'+file
                dfheader = pd.DataFrame(data=data, index=[0])
                dfheader.to_csv(directory+result_filename, index=False)
                n_steps = 1

                for PrH_index in [1, 3, 6, 12, 24, 36, 48, 60, 72]:

                    dataset = pd.read_csv(path+file)

                    # Only the Target
                    dataset = dataset[['year', 'month', 'day', 'hour', target]]

                    # dataset = dataset.dropna()
                    # print(dataset.head())

                    print('Window: '+str(n_steps) + ' TH: ' +
                          str(PrH_index)+' '+method+' '+target)

                    train_X_grid, train_y_grid, input_dim, features = func.preparedata(
                        dataset, PrH_index, n_steps, target, cat)
                    # print(train_y_grid[0:1])

                    start_time = time.time()

                    i = 1
                    # For Test files: #4
                    custom_cv = func.custom_cv_kfolds_testdataonly(
                        train_X_grid, 100)
                    for test_index in custom_cv:

                        # For Train files:
                        # custom_cv = func.custom_cv_2folds(train_X_grid, 3)
                        # for train_index, test_index in custom_cv:
                        test_X = train_X_grid[test_index]
                        test_y = train_y_grid[test_index]

                        # current value would be the same in the future predictions
                        predictions = test_y

                        # actual future values
                        test_y = test_X[:, -1]

                        df_time = pd.DataFrame({
                            'year': np.array(test_X[:, 0]).astype(int), 'month': np.array(test_X[:, 1]).astype(int),
                            'day': np.array(test_X[:, 2]).astype(int), 'hour': np.array(test_X[:, 3]).astype(int),
                        })
                        # print(df_time.head())

                        timeline = pd.to_datetime(df_time, format='%Y%m%d %H')
                        # print(timeline.head())

                        # timeline = timeline.reshape(len(time),)

                        if cat == 1:
                            predictions = np.array(predictions).astype(int)
                            test_y = np.array(test_y).astype(int)

                        test_y = test_y.reshape(len(test_y),)
                        predictions = predictions.reshape(len(predictions),)

                        cm0 = func.forecast_accuracy(predictions, test_y, cat)

                        filename = file + '_' + \
                            target+'_TH' + \
                            str(PrH_index)+'_lag' + \
                            str(n_steps)+'_'+str(i)

                        # First test files
                        if i % 10 == 0:  # or i <= 3:  # 5
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

                            # plt.show()

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

                        elapsed_time = time.time() - start_time
                        print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                        i = i + 1
                    gc.collect()


if __name__ == "__main__":
    main()
