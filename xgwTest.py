# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/xgw/program/graduationDesign/design/programs')
sys.path.append('/home/xgw/program/graduationDesign/design/data')

from sklearn.metrics import accuracy_score,recall_score,confusion_matrix,classification_report  #get accuracy

from xgw.loadData import LoadData
from xgw.xgwModule import  CreateModule,Module
import numpy as np

def accuracy(moduel, test_table, pre_tab):
    print('----------------------',moduel,'----------------------\n')
    print(moduel,'recall_score average:\n',
          recall_score(test_table, pre_tab, average=None))  # the scores for each class are returned
    con_mat = confusion_matrix(test_table, pre_tab)
    print('confusion_matrix:\n', con_mat)
    print(moduel,'accuracy:', accuracy_score(test_table, pre_tab))  # get accuracy
    print('---------------------------------------------------------\n')

if __name__ == '__main__' :


    loadData = LoadData('train_validate_selectData')
    # data = loadData.load()
    #
    # train_input, trian_table, test_input, test_table = loadData.preprocessing(data,6)
    train, test,train_split, test_split = loadData.load()

    train_input, trian_table, test_input, test_table, rf_train_input, rf_test_input =\
        loadData.preprocessing_train_test(train_split, test_split, False)

    modules=CreateModule()


    '''
    ----------------------------neural network-------------------------------
    '''
    nn = modules.neuralNetwork(train_input, trian_table)
    pre_tab_nn = nn.predict(test_input)
    accuracy('neural network', test_table, pre_tab_nn)

    '''
    ----------------------------SVM-------------------------------
    '''

    svm = modules.bagging_svm(train_input, trian_table)
    pre_tab_svm = svm.predict(test_input)
    accuracy('svm',test_table, pre_tab_svm)

    #
    # print('===================SVM=====================')
    # pre_tab=pre_tab_svm
    # print('recall_score average:',
    #       recall_score(test_table, pre_tab, average=None))  # the scores for each class are returned
    # # con_mat = confusion_matrix(test_table, pre_tab)
    # # print('confusion_matrix:\n', con_mat)
    # print('accuracy:', accuracy_score(test_table, pre_tab))  # get accuracy
    # print('===========================================\n')


    # '''
    # -----------------------random forests-------------------------
    # '''
    #
    # rf=modules.randomForests(rf_train_input, trian_table)
    # pre_tab_rf=rf.predict(rf_test_input)
    #
    #
    # # savemodule=Module()
    # # savemodule.save(svm,'svm')
    #
    # print('==============Random Forests================')
    # pre_tab=pre_tab_rf
    # print('recall_score average:',
    #       recall_score(test_table, pre_tab, average=None))  # the scores for each class are returned
    # # con_mat = confusion_matrix(test_table, pre_tab)
    # # print('confusion_matrix:\n', con_mat)
    # print('accuracy:', accuracy_score(test_table, pre_tab))  # get accuracy
    # print('===========================================\n')
