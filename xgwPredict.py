# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/xgw/program/graduationDesign/design/programs')
sys.path.append('/home/xgw/program/graduationDesign/design/data')


from xgw.loadData import LoadData
from xgw.xgwModule import  CreateModule,Module

import numpy as np
from sklearn.metrics import accuracy_score,recall_score,confusion_matrix\
    ,classification_report,cohen_kappa_score,hamming_loss,classification_report,roc_curve
def accuracy(moduel, test_table, pre_tab):
    print('----------------------',moduel,'----------------------\n')
    print(moduel,'recall_score average:\n',
          recall_score(test_table, pre_tab, average=None))  # the scores for each class are returned
    con_mat = confusion_matrix(test_table, pre_tab)
    print('confusion_matrix:\n', con_mat)
    print(moduel,'accuracy:', accuracy_score(test_table, pre_tab))  # get accuracy
    print('---------------------------------------------------------\n')

def analysis(moduel, test_table, pre_tab):
    print('----------------------',moduel,'----------------------\n')
    print(moduel,'recall_score average:\n',
          recall_score(test_table, pre_tab, average=None))  # the scores for each class are returned
    con_mat = confusion_matrix(test_table, pre_tab)
    print('confusion_matrix:\n', con_mat)
    print(moduel,'accuracy:', accuracy_score(test_table, pre_tab))  # get accuracy
    print('---------------------------------------------------------\n')
    # print('==kappa score==:',cohen_kappa_score(test_table, pre_tab))#吻合率,越接近1，两者越一致、越吻合
    # print('==hamming_loss==:',hamming_loss(test_table, pre_tab))#我们可以通过对所有样本的预测情况求平均得到算法在测试集上的总体表现情况
    behavior=['run','work','lie','stand','standing','lying']
    print('==classification report==:\n',classification_report(test_table, pre_tab,target_names=behavior))

def softmax_pro(pro):
    return np.array([np.exp(p)/np.sum(np.exp(pro)) for p in pro])

def vectorized_result(j):
    e = np.zeros((1,6))
    e[0,j] = 1.0
    return e

if __name__ == '__main__' :

    loadData = LoadData('test')
    # data = loadData.load()
    #
    # train_input, trian_table, test_input, test_table = loadData.preprocessing(data,6)
    test = loadData.load()
    test_input, test_table, rf_test_input =\
        loadData.preprocessing_test(test, False)

    savemodule = Module()
    '''
    rf  (neural network and  SVM)
    '''
    rf, rf_probability = savemodule.load('rf')
    svm, svm_probability=savemodule.load('split_svm')
    nn, nn_probability = savemodule.load('split_nn')

    pre_tab_rf_old = rf.predict(rf_test_input)
    pre_probability_rf = rf.predict_proba(rf_test_input)
    # accuracy('random forests', test_table, pre_tab_rf)
    pre_tab_rf=list(pre_tab_rf_old)
    index_i,index_advantage=0,0
    conversion_num=[0,4,5]
    rf_probability_split=[rf_probability[0],rf_probability[4],rf_probability[5]]
    for i in pre_tab_rf:
        if i in (0,4,5):

            need_pre = [test_input[index_i]]
            d_value=np.sort(need_pre)
            if d_value[0,2]-d_value[0,1]<0.1:

                pre_tab_svm = svm.predict(need_pre)
                pre_probability_svm = svm.decision_function(need_pre)
                pre_tab_svm_matrix = np.array(vectorized_result(int(pre_tab_svm)))

                pre_probability_rf_split = [pre_probability_rf[index_i][0],
                                            pre_probability_rf[index_i][4], pre_probability_rf[index_i][5]]
                pre_tab_rf_matrix = np.array(vectorized_result(i))

                pre_tab_nn = nn.predict(test_input[index_i].reshape(1,-1))
                pre_probability_nn = nn.predict_proba(test_input[index_i].reshape(1,-1))
                pre_tab_nn_matrix = np.array(vectorized_result(int(pre_tab_nn)))
                pre_pro = pre_probability_svm
                pre_pro=(np.abs(pre_probability_svm*softmax_pro(svm_probability))+
                         np.abs(pre_probability_rf_split*softmax_pro(rf_probability_split))+
                                np.abs(pre_probability_nn*softmax_pro(nn_probability)))
                pre_tab_rf[index_i]=conversion_num[np.argmax(pre_pro)]
                # pre_pro=(pre_tab_svm_matrix*softmax_pro([svm_probability[0],0,0,0,svm_probability[1],svm_probability[2]])+
                #          pre_tab_rf_matrix*softmax_pro(rf_probability)+
                #          pre_tab_nn_matrix*softmax_pro([nn_probability[0],0,0,0,nn_probability[1],nn_probability[2]]))

                # pre_pro=(pre_tab_svm_matrix+
                #          pre_tab_rf_matrix+
                #          pre_tab_nn_matrix
                #          )
                # d_value=np.sort(pre_pro)
                # if d_value[0,2]-d_value[0,1]<0.1:
                #     if pre_tab_svm==pre_tab_nn :
                #         pre_tab_rf[index_i] = pre_tab_nn
                    # pre_pro=(pre_tab_svm_matrix+
                    #          pre_tab_rf_matrix+
                    #          pre_tab_nn_matrix
                    #          )
                    # pre_tab_rf[index_i] = np.argmax(pre_pro)
                    # d_value=np.sort(pre_pro)
                    # if d_value[0,5]==d_value[0,4]==d_value[0,3]:
                    #     pre_tab_rf[index_i] = pre_tab_svm

                if pre_tab_rf_old[index_i]==test_table[index_i] and pre_tab_rf[index_i]!=test_table[index_i]:
                    print(index_i,':',pre_pro, pre_tab_rf[index_i], pre_tab_rf_old[index_i], test_table[index_i])

                if pre_tab_rf_old[index_i] != test_table[index_i] and pre_tab_rf[index_i] == test_table[index_i]:
                    index_advantage +=1
                # pre_tab_rf[index_i]=np.argmax(pre_pro)
        index_i += 1

    print('advantage:',index_advantage)
    accuracy('old', test_table, pre_tab_rf_old)
    accuracy('==XGW==', test_table, pre_tab_rf)
    # accuracy('==XGW==', pre_tab_rf_old, pre_tab_rf)

    analysis('====', test_table, pre_tab_rf_old)


    '''
    未切分时 总的测试
    ---------------------- svm ----------------------
    svm recall_score average:
     [0.86       0.96666667 1.         1.         0.55       0.4       ]
    confusion_matrix:
     [[344  45   0   1   9   1]
     [  0 290   0  10   0   0]
     [  0   0 240   0   0   0]
     [  0   0   0 400   0   0]
     [  4   1   0   4  11   0]
     [  3   0   3   2   4   8]]
    svm accuracy: 0.9369565217391305
    ---------------------------------------------------------
    ---------------------- random forests ----------------------
    random forests recall_score average:
     [0.9175 1.     1.     0.995  0.55   0.6   ]
    confusion_matrix:
     [[367  26   0   0   6   1]
     [  0 300   0   0   0   0]
     [  0   0 240   0   0   0]
     [  0   2   0 398   0   0]
     [  4   1   0   4  11   0]
     [  1   0   2   2   3  12]]
    random forests accuracy: 0.9623188405797102
    ---------------------------------------------------------
    ---------------------- neural network ----------------------
    neural network recall_score average:
     [0.85       0.94666667 1.         1.         0.55       0.25      ]
    confusion_matrix:
     [[340  45   0   1  14   0]
     [  6 284   0   8   2   0]
     [  0   0 240   0   0   0]
     [  0   0   0 400   0   0]
     [  4   1   0   4  11   0]
     [  3   0   4   2   6   5]]
    neural network accuracy: 0.927536231884058
    ---------------------------------------------------------
    ---------------------- ==XGW== ----------------------
    ==XGW== recall_score average:
     [0.875 0.98  1.    1.    0.65  0.4  ]
    confusion_matrix:
     [[350  40   0   0  10   0]
     [  0 294   0   6   0   0]
     [  0   0 240   0   0   0]
     [  0   0   0 400   0   0]
     [  2   1   0   4  13   0]
     [  3   0   3   2   4   8]]
    ==XGW== accuracy: 0.9456521739130435
---------------------------------------------------------
    '''

    # svm, svm_probability=savemodule.load('svm')
    # pre_tab_svm = svm.predict(test_input)
    # pre_tab_svm_matrix=np.array([vectorized_result(i) for i in pre_tab_svm])
    # pre_probability_svm = svm.decision_function(test_input)
    # accuracy('svm',test_table, pre_tab_svm)
    #
    # rf, rf_probability = savemodule.load('rf')
    # pre_tab_rf = rf.predict(rf_test_input)
    # pre_tab_rf_matrix = np.array([vectorized_result(i) for i in pre_tab_rf])
    # pre_probability_rf = rf.predict_proba(rf_test_input)
    # accuracy('random forests', test_table, pre_tab_rf)
    #
    # nn, nn_probability = savemodule.load('nn')
    # pre_tab_nn = nn.predict(test_input)
    # pre_tab_nn_matrix = np.array([vectorized_result(i) for i in pre_tab_nn])
    # pre_probability_nn = nn.predict_proba(test_input)
    # accuracy('neural network', test_table, pre_tab_nn)
    #
    # # pre_pro=(pre_tab_svm_matrix*softmax_pro(svm_probability)+
    # #          pre_tab_rf_matrix*softmax_pro(rf_probability)+
    # #          pre_tab_nn_matrix*softmax_pro(nn_probability))
    # # pre_pro=(pre_probability_svm*softmax_pro(svm_probability)+
    # #          pre_probability_rf*softmax_pro(rf_probability)+
    # #          pre_probability_nn*softmax_pro(nn_probability))
    # pre_pro=(pre_tab_svm_matrix+\
    #          pre_tab_rf_matrix+\
    #          pre_tab_nn_matrix)
    # pre = np.array([np.argmax(y) for y in pre_pro])
    # accuracy('==XGW==', test_table, pre)




