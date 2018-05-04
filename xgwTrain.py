# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/xgw/program/graduationDesign/design/programs')
sys.path.append('/home/xgw/program/graduationDesign/design/data')

from sklearn.metrics import accuracy_score,recall_score,confusion_matrix,classification_report  #get accuracy

from xgw.loadData import LoadData
from xgw.xgwModule import  CreateModule,Module
import numpy as np

def print_out(moduel, probability, accuracy_rate,best_accuracy):
    print('----------------------',moduel,'----------------------\n')
    mean_probability = np.mean(probability,axis=0)
    mean_accuracy = np.mean(accuracy_rate, axis=0)
    print(moduel,'mean_probability:\n',mean_probability)
    print(moduel,'mean_accuracy:',mean_accuracy,'\n')
    print(moduel,'best_accuracy:',best_accuracy,'\n')
    print('---------------------------------------------------------\n')
    return mean_probability

if __name__ == '__main__' :


    loadData = LoadData('train_validate_selectData')
    train, test, train_split, test_split = loadData.load()
    train_input, trian_table, test_input, test_table, rf_train_input, rf_test_input =\
        loadData.preprocessing_train_test(train, test, False)

    split_train_input, split_trian_table, split_test_input, split_test_table, split_rf_train_input, split_rf_test_input =\
        loadData.preprocessing_train_test(train_split, test_split, False)

    modules=CreateModule()


    svm_accuracy_rate=[]
    svm_probability = []
    svm_best_accuracy=0

    rf_accuracy_rate = []
    rf_probability = []
    rf_best_accuracy = 0

    '''===================second-start====================='''
    for i in range(10) :
        '''
        -----------------------random forests-------------------------
        '''
        rf = modules.randomForests(train_input, trian_table)
        pre_tab_rf = rf.predict(test_input)
        pre_probability_rf = rf.predict_proba(test_input)
        rf_probability.append(recall_score(test_table, pre_tab_rf, average=None))
        rf_as=accuracy_score(test_table, pre_tab_rf)
        rf_accuracy_rate.append(rf_as)
        if rf_as > rf_best_accuracy :
            rf_best_accuracy, rf_module ,pre_probability_best_rf= rf_as, rf,pre_probability_rf

        '''
        ----------------------------SVM-------------------------------
        '''
        svm=modules.bagging_svm(pre_probability_best_rf, trian_table)
        pre_tab_svm=svm.predict(pre_probability_best_rf)
        pre_probability_svm = svm.decision_function(pre_probability_best_rf)
        svm_probability.append(recall_score(test_table, pre_tab_svm, average=None))
        svm_as=accuracy_score(test_table, pre_tab_svm)
        svm_accuracy_rate.append(svm_as)
        if svm_as > svm_best_accuracy :
            svm_best_accuracy, svm_module ,pre_probability_best_svm= svm_as, svm,pre_probability_svm



    rf_mean_probability = np.mean(rf_probability,axis=0)
    rf_mean_accuracy = np.mean(rf_accuracy_rate, axis=0)
    print('rf_mean_probability:\n',rf_mean_probability)
    print('rf_mean_accuracy:',rf_mean_accuracy,'\n')
    print('rf_best_accuracy:',rf_best_accuracy,'\n')

    svm_mean_probability=np.mean(svm_probability,axis=0)
    svm_mean_accuracy = np.mean(svm_accuracy_rate, axis=0)
    print('svm_mean_probability:\n',svm_mean_probability)
    print('svm_mean_accuracy:',svm_mean_accuracy,'\n')
    print('svm_best_accuracy:',svm_best_accuracy,'\n')



    '''
    -----------------------neural network-------------------------
    '''
    nn_accuracy_rate=[]
    nn_probability = []
    nn_best_accuracy=0
    for i in range(6) :
        nn = modules.neuralNetwork(pre_probability_best_svm, trian_table)
        pre_tab_nn = nn.predict(pre_probability_best_svm)
        nn_probability.append(recall_score(test_table, pre_tab_nn, average=None))
        nn_as=accuracy_score(test_table, pre_tab_nn)
        nn_accuracy_rate.append(nn_as)
        if nn_as > nn_best_accuracy :
            nn_best_accuracy, nn_module = nn_as, nn

    nn_mean_probability=print_out('neural network',nn_probability, nn_accuracy_rate,nn_best_accuracy)

    # savemodule=Module()
    # savemodule.save(svm_module,'split_svm_stacking',svm_mean_probability)
    # savemodule.save(rf_module, 'rf', rf_mean_probability)
    # savemodule.save(nn_module, 'split_nn_stacking', nn_mean_probability)

    '''===================second-end====================='''


    '''
    first-start
    '''
    # for i in range(10) :
    #     '''
    #     ----------------------------SVM-------------------------------
    #     '''
    #     svm=modules.bagging_svm(split_train_input, split_trian_table)
    #     pre_tab_svm=svm.predict(split_test_input)
    #     svm_probability.append(recall_score(split_test_table, pre_tab_svm, average=None))
    #     svm_as=accuracy_score(split_test_table, pre_tab_svm)
    #     svm_accuracy_rate.append(svm_as)
    #     if svm_as > svm_best_accuracy :
    #         svm_best_accuracy, svm_module = svm_as, svm
    #
    #
    #     '''
    #     -----------------------random forests-------------------------
    #     '''
    #     rf = modules.randomForests(rf_train_input, trian_table)
    #     pre_tab_rf = rf.predict(rf_test_input)
    #     rf_probability.append(recall_score(test_table, pre_tab_rf, average=None))
    #     rf_as=accuracy_score(test_table, pre_tab_rf)
    #     rf_accuracy_rate.append(rf_as)
    #     if rf_as > rf_best_accuracy :
    #         rf_best_accuracy, rf_module = rf_as, rf
    #
    #
    # svm_mean_probability=np.mean(svm_probability,axis=0)
    # svm_mean_accuracy = np.mean(svm_accuracy_rate, axis=0)
    # print('svm_mean_probability:\n',svm_mean_probability)
    # print('svm_mean_accuracy:',svm_mean_accuracy,'\n')
    # print('svm_best_accuracy:',svm_best_accuracy,'\n')
    #
    #
    # rf_mean_probability = np.mean(rf_probability,axis=0)
    # rf_mean_accuracy = np.mean(rf_accuracy_rate, axis=0)
    # print('rf_mean_probability:\n',rf_mean_probability)
    # print('rf_mean_accuracy:',rf_mean_accuracy,'\n')
    # print('rf_best_accuracy:',rf_best_accuracy,'\n')
    #
    # '''
    # -----------------------neural network-------------------------
    # '''
    # nn_accuracy_rate=[]
    # nn_probability = []
    # nn_best_accuracy=0
    # for i in range(6) :
    #     nn = modules.neuralNetwork(split_train_input, split_trian_table)
    #     pre_tab_nn = nn.predict(split_test_input)
    #     nn_probability.append(recall_score(split_test_table, pre_tab_nn, average=None))
    #     nn_as=accuracy_score(split_test_table, pre_tab_nn)
    #     nn_accuracy_rate.append(nn_as)
    #     if nn_as > nn_best_accuracy :
    #         nn_best_accuracy, nn_module = nn_as, nn
    #
    # nn_mean_probability=print_out('neural network',nn_probability, nn_accuracy_rate,nn_best_accuracy)
    #
    # # savemodule=Module()
    # # savemodule.save(svm_module,'split_svm',svm_mean_probability)
    # # savemodule.save(rf_module, 'rf', rf_mean_probability)
    # # savemodule.save(nn_module, 'split_nn', nn_mean_probability)

'''
first -end 
'''