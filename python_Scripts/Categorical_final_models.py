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


def main():

    models = ['RF']  # 'LSTM', 'NN', 'LR', 'RF', 'DT', 'SVC',
    targets = ['dissolved_oxygen', 'ph']  # ['DOcategory', 'pHcategory']
    sondefilename = 'leavon_wo_2019-07-01-2020-01-15'
    n_job = -1

    for model_name in models:
        print(model_name)

        for target in targets:
            if target.find('category') > 0:
                cat = 1
                directory = 'Results/balance_data/output_Cat_' + model_name+'/final_models/'
                data = {'target_names': 'target_names', 'method_names': 'method_names', 'temporalhorizons': 'temporalhorizons', 'CV': 'CV',
                        'file_names': 'file_names', 'F1_0': 'F1_0', 'F1_1': 'F1_1', 'P_0': 'P_0', 'P_1': 'P_1', 'R_0': 'R_0', 'R_1': 'R_1', 'acc0_1': 'acc0_1', 'F1_0_1': 'F1_0_1'}
            else:
                cat = 0
                directory = 'Results/balance_data/output_Reg_' + model_name+'/final_models/'
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

            if model_name == 'DT' or model_name == 'RF':
                method = 'OrgData'
                path = 'Sondes_data/train/train_data/'
                testpath = 'Sondes_data/test/test_data/'
            else:
                method = 'StandardScaler'
                path = 'Sondes_data/train/train_data_normalized/'+method+'/'+target+'/'
                testpath = 'Sondes_data/test/train_data_normalized/' + method+'/'+target+'/'

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

                if model_name == 'LSTM' or model_name == 'NN':
                    n_job = 1

                start_time = time.time()

                clf = func.getModel(
                    model_name, input_dim, params['param_'+target+'_'+str(PrH_index)], n_job, cat)

                print('clf: '+str(clf))

                if cat == 1 and (model_name == 'LSTM' or model_name == 'NN'):
                    train_y_grid = to_categorical(train_y_grid, 3)
                    clf = clf.fit(train_X_grid, train_y_grid,
                                  model__class_weight={0: 1, 1: 50, 2: 100})
                else:
                    clf = clf.fit(train_X_grid, train_y_grid)

                # save the model to disk
                filename = model_name+'_model_' + \
                    target+'_'+str(PrH_index)+'.sav'
                joblib.dump(clf, directory+filename)

                # if model_name == 'RF' or model_name=='DT':
                #     featurenames = func.setfeatures(features, lags)

                #     if not os.path.exists(directory+'trees/'):
                #         os.makedirs(directory+'trees/')

                #     i_tree = 0
                #     class_names = ['0', '1', '2']
                #     print(len(clf))
                #     for tree_in_forest in clf:
                #         dot_data = tree.export_graphviz(tree_in_forest, out_file=None,
                #                                         feature_names=featurenames,
                #                                         class_names=class_names,
                #                                         filled=True, rounded=True,
                #                                         special_characters=True)
                #         graph = pydotplus.graph_from_dot_data(dot_data)
                #         graph.write_pdf(
                #             directory+'trees/tree_'+filename+str(i_tree)+".pdf")
                #         i_tree = i_tree + 1
                #         if(i_tree > 1):
                #             break

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

                    # test_y = test_y.astype(int)
                    # predictions = predictions.astype(int)

                    if i % 10 == 0:
                        plt.scatter(np.arange(len(test_y)),
                                    test_y, s=1)
                        plt.scatter(np.arange(len(predictions)),
                                    predictions, s=1)
                        plt.legend(['actual', 'predictions'],
                                   loc='upper right')
                        fpath = 'predictions_' + method+target+'_Window' + \
                            str(lags) + '_TH'+str(PrH_index) + \
                            '_CV' + str(i)+file1
                        plt.savefig(directoryresult+fpath+'.jpg')

                        plt.close()
                    #     data = {'Actual': test_y, 'Predictions': predictions}
                    #     print(test_y.shape)
                    #     print(predictions.shape)
                    #     if model_name == 'RF':
                    #         df = pd.DataFrame(data=data)
                    #     else:
                    #         df = pd.DataFrame(data=data, index=[0])

                    #     df.to_csv(directoryresult+filename +
                    #             '_CV'+str(i)+'.csv', index=False)

                    cm0 = func.forecast_accuracy(predictions, test_y, cat)

                    if cat == 1:
                        data = {'target_names': target, 'method_names': method, 'temporalhorizons': PrH_index, 'CV': i,
                                'file_names': filename,  'F1_0': cm0[0], 'F1_1': cm0[1], 'P_0': cm0[2], 'P_1': cm0[3], 'R_0': cm0[4], 'R_1': cm0[5], 'acc0_1': cm0[6], 'F1_0_1': cm0[7]}
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

# thres = dict()
# thres[1] = {0: 0.36507936507936506, 1: 0.24107974869991283, 2: 0.7566056452340157, 'micro': 0.36507936507936506}
# thres[3] = {0: 0.37119884547984744, 1: 0.25020204038421967, 2: 0.7377828938147312, 'micro': 0.37119884547984744}
# thres[6] = {0: 0.37961595273264404, 1: 0.23941720077507825, 2: 0.7474412480747255, 'micro': 0.37961595273264404}
# thres[12] = {0: 0.4401309882216236, 1: 0.29361780544237187, 2: 0.6650422817830013, 'micro': 0.4401309882216236}
# thres[24] = {0: 0.29785661492978566, 1: 0.18451416541607066, 2: 0.5621027753824535, 'micro': 0.3735650383975932}
# thres[36] = {0: 0.3731461340715839, 1: 0.09845367078212343, 2: 0.7188528621694222, 'micro': 0.17868243826627525}
# thres[48] = {0: 0.3145797029292175, 1: 0.12408124229645111, 2: 0.7731371126541611, 'micro': 0.21910216636173202}

# print(thres[PrH_index][0])
# print(thres[PrH_index][1])
# print(thres[PrH_index][2])

# predict_mine0 = np.where(yhat[:,0] >= thres[PrH_index][0],1,0)
# predict_mine1 = np.where(yhat[:,1] >= thres[PrH_index][1],1,0)
# predict_mine2 = np.where(yhat[:,2] >= thres[PrH_index][2],1,0)

# predict_mine = np.concatenate((predict_mine0.reshape(-1,1),predict_mine1.reshape(-1,1),predict_mine2.reshape(-1,1)),axis=1)
# print((predict_mine.shape))
# predictions = argmax(predict_mine, axis=1)
