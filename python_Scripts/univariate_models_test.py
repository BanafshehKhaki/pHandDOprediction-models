from statsmodels.tsa.arima_model import ARIMAResults
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import time
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import re
import datetime
import sklearn.metrics as skm
import functions as func
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")


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


def ARIMAregression(train, config):
    p, d, q = config
    history = [x for x in train]
    model = ARIMA(history, order=(p, d, q))
    model_fit = model.fit(disp=0)
    return model_fit


def movingAverage(train_X, train_y, test_X, test_y):

    # calculate residuals for # persistence model on training set
    train_resid = [train_y[i]-train_X[i] for i in range(len(train_X))]

    # model the training set residuals
    model = AR(train_resid)
    model_fit = model.fit()
    window = model_fit.k_ar
    coef = model_fit.params
    lag = train_resid[-window:]
    return coef, lag


def AutoRegression(train):

    model = AR(train)
    model_fit = model.fit()
    window = model_fit.k_ar
    coef = model_fit.params
    lag = train[-window:]
    return coef, lag


def custom_cv_2folds(X, kfolds, th):
    n = X.shape[0]
    print('******** creating custom CV:')
    i = 1
    while i <= kfolds:
        np.random.seed(i)
        idx = np.empty(0, dtype=int)
        for index in np.arange(0, n-(th*6), step=(th*6), dtype=int):
            randwindowpoint = np.random.randint(0, 6, size=1)
            idx = np.append(idx, [randwindowpoint+index])
            # print(idx)
        print(idx[0: 10])
        yield idx[: int(len(idx)*0.7)], idx[int(len(idx)*0.7):]
        i = i+1


def custom_cv_kfolds_testdataonly(X, kfolds, th):
    n = X.shape[0]
    # print(n)
    print('******** creating custom CV:')
    i = 1
    while i <= kfolds:
        np.random.seed(i)
        idx = np.empty(0, dtype=int)
        for index in np.arange(0, n-(th*6), step=(th*6), dtype=int):
            randwindowpoint = np.random.randint(0, 6, size=1)
            idx = np.append(idx, [randwindowpoint+index])
            # print(idx)
        print(idx[0:10])
        yield idx[:int(len(idx))]
        i = i+1


def predict(coef, lag, window):
    yhat = coef[0]
    for d in range(1, window):
        yhat += coef[d] * lag[-d]
    return yhat


def getconfig(target, PrH_index, model_name):
    if model_name == 'ARIMA':
        if target == 'ph':
            if PrH_index == 1 or PrH_index == 3:
                return (1, 0, 0)
            elif PrH_index == 6 or PrH_index == 12:
                return (1, 1, 0)
            elif PrH_index == 24 or PrH_index == 36 or PrH_index == 72:
                return (2, 1, 0)
            elif PrH_index == 48 or PrH_index == 60:
                return (1, 1, 1)
        else:
            if PrH_index == 1 or PrH_index == 3:
                return (1, 0, 2)
            elif PrH_index == 6 or PrH_index == 12:
                return (2, 0, 0)
            elif PrH_index == 24 or PrH_index == 60:
                return (1, 1, 1)
            elif PrH_index == 36 or PrH_index == 72:
                return (1, 0, 0)
            elif PrH_index == 48:
                return (2, 1, 1)
    if model_name == 'ETS':
        if target == 'ph':
            if PrH_index == 3:
                return ('add', False, None, 3600, False, False)
            elif PrH_index >= 6 and PrH_index != 60:
                return ('mul', True, None, None, False, False)
            elif PrH_index == 60:
                return ('add', True, None, 3600, False, True)
        else:
            if PrH_index < 72:
                return ('add', False, None, 3600, False, False)
            else:
                return ('add', True, None, 3600, False, True)
    if model_name == 'SARIMA':
        if target == 'dissolved_oxygen':
            if PrH_index == 1 or PrH_index == 3:
                return ((0, 0, 0), (1, 0, 2, 0), 'c')
            elif PrH_index == 6:
                return((0, 0, 2), (2, 0, 2, 0), 'c')
            elif PrH_index == 12:
                return((0, 0, 2), (1, 0, 0, 0), 'ct')
            elif PrH_index == 24:
                return((1, 0, 0), (1, 0, 0, 0), 'ct')
            elif PrH_index == 36:
                return((0, 0, 0), (1, 0, 0, 0), 'ct')
            elif PrH_index == 48 or PrH_index == 60:
                return((0, 0, 2), (0, 0, 0, 3600), 'ct')
            elif PrH_index == 72:
                return((0, 0, 1), (0, 0, 0, 3600), 'ct')
        else:
            if PrH_index == 1:
                return ((1, 0, 0), (0, 0, 0, 720), 'n')
            elif PrH_index == 3:
                return((0, 0, 0), (1, 0, 0, 0), 'n')
            elif PrH_index == 6:
                return((0, 1, 0), (0, 0, 0, 0), 'n')
            elif PrH_index == 12:
                return((2, 0, 2), (2, 0, 2, 0), 'c')
            elif PrH_index == 24:
                return((0, 0, 1), (1, 0, 0, 0), 'c')
            elif PrH_index == 36:
                return((1, 0, 2), (0, 0, 0, 720), 'n')
            elif PrH_index == 48:
                return((1, 0, 0), (0, 0, 0, 720), 'c')
            elif PrH_index == 60:
                return((2, 0, 0), (0, 0, 0, 720), 'c')
            elif PrH_index == 72:
                return((1, 0, 1), (0, 0, 0, 720), 'n')


# one-step Holt Winter’s Exponential Smoothing forecast


def ETSregression(history, config):
    t, d, s, p, b, r = config
    # define model
    history = np.array(history)
    model = ExponentialSmoothing(
        history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
    # fit model
    model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
    return model_fit


def SARIMAregression(history, config):
    order, sorder, trend = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend,
                    enforce_stationarity=False, enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False, maxiter=2000)

    return model_fit


def main():

    models = ['SARIMA']
    targets = ['dissolved_oxygen']
    sondefilename = 'leavon'
    n_job = -1

    for model_name in models:
        print(model_name)

        for target in targets:
            if target.find('category') > 0:
                cat = 1
                directory = 'Results/bookThree/1sonde/output_Cat_' + \
                    model_name+'/final_models/'
                data = {'target_names': 'target_names', 'method_names': 'method_names', 'window_nuggets': 'window_nuggets', 'temporalhorizons': 'temporalhorizons', 'CV': 'CV',
                        'file_names': 'file_names', 'F1_0': 'F1_0', 'F1_1': 'F1_1', 'P_0': 'P_0', 'P_1': 'P_1', 'R_0': 'R_0', 'R_1': 'R_1', 'acc0_1': 'acc0_1', 'F1_0_1': 'F1_0_1', 'F1_all': 'F1_all', 'fbeta': 'fbeta'}
            else:
                cat = 0
                directory = 'Results/bookThree/1sonde/output_Reg_' + model_name+'/final_models/'
                data = {'target_names': 'target_names', 'method_names': 'method_names', 'window_nuggets': 'window_nuggets', 'temporalhorizons': 'temporalhorizons', 'CV': 'CV',
                        'file_names': 'file_names',  'mape': 'mape', 'me': 'me', 'mae': 'mae', 'mpe': 'mpe', 'rmse': 'rmse',  'R2': 'R2'}

            if not os.path.exists(directory):
                os.makedirs(directory)

            directoryresult = directory + 'Results/'
            if not os.path.exists(directoryresult):
                os.makedirs(directoryresult)

            resultFileName = 'results_'+target+str(time.time())+'.csv'
            dfheader = pd.DataFrame(data=data, index=[0])
            dfheader.to_csv(directoryresult+resultFileName,
                            index=False, header=False)

            path = 'Sondes_data/train_Summer/'
            testpath = 'Sondes_data/test_Summer/'
            method = 'OrgData'

            for n_steps in [1]:
                for PrH_index in [48, 60]:  # 1, 3, 6, 12, 24, 36,
                    files = [f for f in os.listdir(path) if f.endswith(
                        '.csv') and f.startswith(sondefilename)]
                    file = files[0]
                    print('Window: '+str(n_steps) + ' TH: ' +
                          str(PrH_index)+' '+method+' '+target)

                    dataset = pd.read_csv(path+file)

                    ######################
                    # FOR  ARIMA
                    ######################
                    train = dataset[target]
                    custom_cv = custom_cv_kfolds_testdataonly(
                        train, 1, PrH_index)
                    for train_index in custom_cv:
                        train_y = train[train_index].values

                        config = getconfig(target, PrH_index, model_name)
                        if model_name == 'ARIMA':
                            model_fit = ARIMAregression(train_y, config)
                            model_fit.save(directory+'ARIMA_model'+target +
                                           '_'+str(PrH_index)+'.pkl')
                        elif model_name == 'ETS':
                            model_fit = ETSregression(train_y, config)
                            model_fit.save(directory+'ETS_model'+target +
                                           '_'+str(PrH_index)+'.pkl')

                        elif model_name == 'SARIMA':
                            model_fit = SARIMAregression(train_y, config)
                            model_fit.save(directory+'SARIMA_model'+target +
                                           '_'+str(PrH_index)+'.pkl')

                            ######################
                            # TEST sets
                            ######################
                        start_time = time.time()
                        testsondefilename = sondefilename
                        files = [f for f in os.listdir(testpath) if f.endswith(
                            '.csv')and f.startswith(testsondefilename)]
                        file1 = files[0]

                        testdataset = pd.read_csv(testpath+file1)

                        test = testdataset[target]
                        i = 1
                        custom_cv = custom_cv_kfolds_testdataonly(
                            test, 5, PrH_index)
                        for test_index in custom_cv:
                            test_y = test[test_index].values

                            # ARIMA
                            history = [train_y[i]for i in range(len(train_y))]
                            predictions = list()

                            for t in range(len(test_y)):
                                if model_name == 'ARIMA':
                                    model = ARIMA(history, order=(config))
                                    model_fit = model.fit(disp=0)
                                    yhat, stderr, conf = model_fit.forecast()

                                elif model_name == 'ETS':
                                    model_fit = ETSregression(history, config)
                                    yhat = model_fit.forecast()

                                elif model_name == 'SARIMA':
                                    model_fit = SARIMAregression(
                                        history, config)
                                    yhat = model_fit.forecast()

                                predictions.append(yhat)
                                history.append(test_y[t])

                            if cat == 1:
                                predictions = np.array(
                                    predictions).astype(int)

                            fpath = 'predictions_' + method+target+'_Window' + \
                                str(n_steps) + '_TH' + \
                                str(PrH_index)+'_CV' + str(i) + file

                            cm0 = func.forecast_accuracy(
                                predictions, test_y, cat)

                            if i % 10 == 0 or i <= 5:
                                plt.scatter(np.arange(len(test_y)),
                                            test_y, s=1)
                                plt.scatter(np.arange(len(predictions)),
                                            predictions, s=1)
                                plt.legend(['actual', 'predictions'],
                                           loc='upper right')
                                plt.savefig(directoryresult+fpath+'.png')
                                plt.close()

                                data = {'Actual': test_y,
                                        'Predictions': predictions}
                                df = pd.DataFrame(data=data)
                                df.to_csv(directoryresult +
                                          fpath, index=False)

                            if cat == 1:
                                data = {'target_names': target, 'method_names': method, 'window_nuggets': n_steps, 'temporalhorizons': PrH_index, 'CV': i,
                                        'file_names': fpath, 'F1_0': cm0[0], 'F1_1': cm0[1], 'P_0': cm0[2], 'P_1': cm0[3], 'R_0': cm0[4], 'R_1': cm0[5], 'acc0_1': cm0[6], 'F1_0_1': cm0[7], 'F1_all': cm0[8], 'fbeta': [cm0[9]]}
                            elif cat == 0:
                                data = {'target_names': target, 'method_names': method, 'window_nuggets': n_steps, 'temporalhorizons': PrH_index, 'CV': i,
                                        'file_names': fpath,  'mape': cm0[0], 'me': cm0[1], 'mae': cm0[2], 'mpe': cm0[3], 'rmse': cm0[4], 'R2': cm0[5]}

                            df = pd.DataFrame(data=data, index=[0])
                            df.to_csv(directoryresult+resultFileName,
                                      index=False, mode='a', header=False)

                            i = i+1


if __name__ == "__main__":
    main()
