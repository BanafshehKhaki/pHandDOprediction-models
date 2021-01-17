from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree
import pandas as pd
import numpy as np
import pydotplus
import joblib
import time
import re
import os
import pickle
from numpy import argmax
import functions as func
from keras import backend as Kb
from keras.utils import to_categorical
import gc
from imblearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


def R2_measure(y_true, y_pred):
    return r2_score(y_true, y_pred)


def f2_measure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, labels=[1, 2],  beta=2, average='micro')


def main():

    models = ['NN']  # 'LSTM', 'NN', 'LR', 'RF', 'DT', 'SVC',
    # 'DOcategory', 'pHcategory','ph', 'dissolved_oxygen',
    targets = ['pHcategory']
    sondefilename = 'leavon_wo_2019-07-01-2020-01-15'
    n_job = -1

    for model_name in models:
        print(model_name)

        for target in targets:
            if target.find('category') > 0:
                cat = 1
                directory = 'Results/bookOne/output_Cat_' + model_name+'/final_models/'
                data = {'target_names': 'target_names', 'method_names': 'method_names', 'temporalhorizons': 'temporalhorizons', 'CV': 'CV',
                        'file_names': 'file_names', 'F1_0': 'F1_0', 'F1_1': 'F1_1', 'P_0': 'P_0', 'P_1': 'P_1', 'R_0': 'R_0', 'R_1': 'R_1', 'acc0_1': 'acc0_1', 'F1_0_1': 'F1_0_1', 'F1_all': 'F1_all', 'fbeta': 'fbeta'}
            else:
                cat = 0
                directory = 'Results/bookOne/output_Reg_' + model_name+'/final_models/'
                data = {'target_names': 'target_names', 'method_names': 'method_names', 'temporalhorizons': 'temporalhorizons', 'CV': 'CV',
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

            path = 'Sondes_data/train/train_data/'
            testpath = 'Sondes_data/test/test_data/'
            method = 'OrgData'

            for PrH_index in [1, 3, 6, 12, 24, 36, 48]:
                params = func.trained_param_grid[
                    'param_grid_'+model_name+str(cat)]
                lags = func.getlags_window(
                    model_name, params['param_'+target+'_'+str(PrH_index)], cat)

                files = [f for f in os.listdir(path) if f.endswith(
                    '.csv') and f.startswith(sondefilename)]
                file1 = files[0]
                print(' TH: ' +
                      str(PrH_index)+' '+method+' '+target+' '+file1)

                dataset = pd.read_csv(path+file1)
                train_X_grid, train_y_grid, input_dim, features = func.preparedata(
                    dataset, PrH_index, lags, target, cat)
                print(input_dim)

                if cat == 1 and (model_name == 'LSTM' or model_name == 'NN'):
                    train_y_grid = to_categorical(train_y_grid, 3)

                start_time = time.time()

                mo = func.getModel(
                    model_name, input_dim, params['param_'+target+'_'+str(PrH_index)], n_job, cat)

                if model_name == 'RF' or model_name == 'DT':
                    pipeline = Pipeline(steps=[('model', mo)])
                else:
                    pipeline = Pipeline(
                        steps=[('n', StandardScaler()), ('model', mo)])

                # save the model to disk
                filename = model_name+'_model_' + \
                    target+'_'+str(PrH_index)+'.sav'

                if cat == 1 and (model_name == 'LSTM' or model_name == 'NN'):
                    clf = pipeline.fit(train_X_grid, train_y_grid, model__class_weight={
                                       0: 1, 1: 50, 2: 100})
                else:
                    clf = pipeline.fit(train_X_grid, train_y_grid)

                # joblib.dump(clf, directory+filename)
                pickle.dump(clf, open(directory+filename, 'wb'))

                # To load the model, open the file in reading and binary mode
                # load_lr_model =pickle.load(open(filename, 'rb'))

                elapsed_time = time.time() - start_time
                print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

                #################################
                # Testing final model on test data
                #################################
                start_time = time.time()
                testsondefilename = re.sub('wo_', '', sondefilename)
                files = [f for f in os.listdir(testpath) if f.endswith(
                    '.csv')and f.startswith(testsondefilename)]
                file1 = files[0]
                print('Window: '+str(lags) + ' TH: ' +
                      str(PrH_index)+' '+method+' '+target+file1)

                dataset = pd.read_csv(testpath+file1)

                test_X_grid, test_y_grid, input_dim, features = func.preparedata(
                    dataset, PrH_index, lags, target, cat)

                if cat == 1 and (model_name == 'LSTM' or model_name == 'NN'):
                    test_y_grid = to_categorical(test_y_grid, 3)

                i = 1
                custom_cv = func.custom_cv_kfolds_testdataonly(
                    test_X_grid, 100)
                for test_index in custom_cv:
                    test_X = test_X_grid[test_index]
                    test_y = test_y_grid[test_index]

                    predictions = clf.predict(test_X)

                    if model_name == 'LSTM' or model_name == 'NN':
                        test_y = argmax(test_y, axis=1)
                        # predictions = argmax(predictions, axis=1)

                    if cat == 1:
                        predictions = np.array(predictions).astype(int)
                        test_y = np.array(test_y).astype(int)
                        test_y = test_y.reshape(len(test_y),)
                        predictions = predictions.reshape(len(predictions),)

                    if i % 10 == 0:
                        plt.scatter(np.arange(len(test_y)),
                                    test_y, s=1)
                        plt.scatter(np.arange(len(predictions)),
                                    predictions, s=1)
                        plt.legend(['actual', 'predictions'],
                                   loc='upper right')
                        fpath = filename + '_CV'+str(i) + file1
                        # 'predictions_' + method+target+'_Window' + str(lags) + '_TH'+str(PrH_index) + \'_CV' + str(i)+file1
                        plt.savefig(directoryresult+fpath+'.jpg')

                        plt.close()
                        data = {'Actual': test_y, 'Predictions': predictions}
                        print(test_y.shape)
                        print(predictions.shape)
                        df = pd.DataFrame(data=data)
                        df.to_csv(directoryresult+filename +
                                  '_CV'+str(i) + file1, index=False)

                    cm0 = func.forecast_accuracy(predictions, test_y, cat)

                    if cat == 1:
                        data = {'target_names': target, 'method_names': method, 'temporalhorizons': PrH_index, 'CV': i,
                                'file_names': filename,  'F1_0': cm0[0], 'F1_1': cm0[1], 'P_0': cm0[2], 'P_1': cm0[3], 'R_0': cm0[4], 'R_1': cm0[5], 'acc0_1': cm0[6], 'F1_0_1': cm0[7], 'F1_all': cm0[8], 'fbeta': [cm0[9]]}
                    elif cat == 0:
                        data = {'target_names': target, 'method_names': method, 'temporalhorizons': PrH_index, 'CV': i,
                                'file_names': filename, 'mape': cm0[0], 'me': cm0[1], 'mae': cm0[2], 'mpe': cm0[3], 'rmse': cm0[4], 'R2': cm0[5]}

                    df = pd.DataFrame(data=data, index=[0])
                    df.to_csv(directoryresult+resultFileName,
                              index=False, mode='a', header=False)

                    elapsed_time = time.time() - start_time
                    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                    i = i+1
                Kb.clear_session()
                gc.collect()
                del clf


if __name__ == "__main__":
    main()
