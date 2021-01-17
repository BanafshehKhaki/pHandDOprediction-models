import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objs as go
import os
import functions as func
import time


models = ['baseline', 'MA', 'AR', 'ARIMA', 'ETS']
folder = 'bookThree/1sonde'

for model in models:
    for target in ['ph', 'dissolved_oxygen']:
        for th in [1, 3, 6, 12, 24, 36, 48, 60, 72]:
            # print(model)
            testdirectory = 'Results/'+folder+'/output_Reg' + \
                '_' + model + '/final_models/Results/'

            data = {'target_names': 'target_names',  'CV': 'CV', 'TH': 'TH',
                    'file_names': 'file_names',  'mape': 'mape', 'me': 'me', 'mae': 'mae', 'mse': 'mse', 'rmse': 'rmse',  'R2': 'R2'}

            result_filename = 'results_alternative_final_'+target + \
                '_'+str(time.time())+'.csv'
            dfheader = pd.DataFrame(data=data, index=[0])
            dfheader.to_csv(testdirectory+result_filename, index=False)

            files = [f for f in os.listdir(testdirectory) if f.endswith(
                ".csv") and f.startswith('leavon.csv_'+target+'_TH'+str(th))]
            i_cv = 10
            for file in files:
                results = pd.read_csv(testdirectory+file)

                if target == 'ph':
                    indexNames = results[results['Actual'] < 8].index
                    results.drop(indexNames, inplace=True)

                else:
                    indexNames = results[results['Actual'] < 4].index
                    results.drop(indexNames, inplace=True)

                results = results.reset_index()

                test_y = results[['Actual']].values
                predictions = results[['Predictions']].values
                # print(predictions)
                # print(test_y)

                cm0 = func.forecast_accuracy(predictions, test_y, 0)

                data = {'target_names': target, 'CV': i_cv, 'TH': th,
                        'file_names': file,  'mape': cm0[0], 'me': cm0[1], 'mae': cm0[2], 'mpe': cm0[3], 'rmse': cm0[4], 'R2': cm0[5]}
                i_cv = i_cv + 10
                df = pd.DataFrame(data=data, index=[0])
                df.to_csv(testdirectory+result_filename,
                          index=False, mode='a', header=False)


models = ['endecodeLSTM', 'CNNLSTM', 'ConvEnLSTM',
          'NN', 'LSTM_0', 'SVC', 'RF_onereg', 'RF', 'DT_onereg', 'DT']
folder = 'bookThree/2sondes'

for model in models:
    for target in ['ph', 'dissolved_oxygen']:
        print(model)
        testdirectory = 'Results/'+folder+'/output_Reg' + \
            '_' + model + '/final_models/Results/'

        data = {'target_names': 'target_names',  'CV': 'CV',
                'file_names': 'file_names',  'mape': 'mape', 'me': 'me', 'mae': 'mae', 'mse': 'mse', 'rmse': 'rmse',  'R2': 'R2'}

        result_filename = 'results_alternative_final_'+target + \
            '_'+str(time.time())+'.csv'
        dfheader = pd.DataFrame(data=data, index=[0])
        dfheader.to_csv(testdirectory+result_filename, index=False)

        files = [f for f in os.listdir(testdirectory) if f.endswith(
            ".csv") and f.startswith('predictions_OrgData'+target)]
        i_cv = 10
        for file in files:
            results = pd.read_csv(testdirectory+file)
            s = pd.DataFrame()
            if target == 'ph':
                # indexNames = results[(
                #     results['a+12'] > 8) & (results['a+36'] > 8) & (results['a+72'] > 8)].index
                # results.drop(indexNames, inplace=True)
                for item in [1, 3, 6, 12, 24, 36, 48, 60, 72]:
                    s = results[(results['a+'+str(item)] < 8)]
                    # s = results.drop(indexNames)
                    results['a+'+str(item)] = s['a+'+str(item)]
                    results['p+'+str(item)] = s['p+'+str(item)]

            else:
                for item in [1, 3, 6, 12, 24, 36, 48, 60, 72]:
                    s = results[(results['a+'+str(item)] < 4)]
                    # s = results.drop(indexNames)
                    results['a+'+str(item)] = s['a+'+str(item)]
                    results['p+'+str(item)] = s['p+'+str(item)]
                # indexNames = results[(
                #     results['a+12'] > 4) & (results['a+36'] > 4) & (results['a+72'] > 4)].index
                # results.drop(indexNames, inplace=True)

            # results = results.reset_index()

            test_y = results[['a+1', 'a+3', 'a+6', 'a+12',
                              'a+24', 'a+36', 'a+48', 'a+60', 'a+72']].values
            predictions = results[['p+1', 'p+3', 'p+6',
                                   'p+12', 'p+24', 'p+36', 'p+48', 'p+60', 'p+72']].values

            # print(predictions[0:10])
            # print(test_y[0:10])

            cm0 = np.zeros((9, 6))
            for t in range(9):
                y = test_y[:, t]
                y = y[~np.isnan(y)]
                yhat = predictions[:, t]
                yhat = yhat[~np.isnan(yhat)]
                cm0[t, :] = func.forecast_accuracy(yhat, y, 0)

            data = {'target_names': target, 'CV': i_cv,
                    'file_names': file,  'mape': [cm0[:, 0]], 'me': [cm0[:, 1]], 'mae': [cm0[:, 2]], 'mse': [cm0[:, 3]], 'rmse': [cm0[:, 4]], 'R2': [cm0[:, 5]]}
            i_cv = i_cv + 10
            df = pd.DataFrame(data=data, index=[0])
            df.to_csv(testdirectory+result_filename,
                      index=False, mode='a', header=False)
