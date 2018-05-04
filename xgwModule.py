# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn import tree
from sklearn import svm
from sknn.mlp import Classifier,Layer

import pickle


class CreateModule(object) :

    def svm(self,train_input, trian_table):

        clf = svm.SVC(C=300.0, cache_size=200, class_weight=None, coef0=0.0,  # C的大小表示对错误的容忍程度，很小表示很大的错误容忍
                      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                      # decision_function_shape='ovr'#gamma值表示模型的复杂程度，越大映射维度就越大，模型越复杂，但是越复杂的泛化能力越低、使用价值也偏低
                      max_iter=-1, probability=False, random_state=None, shrinking=True,
                      tol=0.001, verbose=False)
        clf.fit(train_input, trian_table)  # 对过采样数据进行训练，未处理数据选部分进行预测


    def randomForests(self,train_input, trian_table):

        decisionTree = tree.DecisionTreeClassifier(criterion='gini', max_depth=30)  # 熵、另可用gini
        clf = RandomForestClassifier(n_estimators=38)
        clf.fit(train_input, trian_table)  # 对过采样数据进行训练，未处理数据选部分进行预测

    def neuralNetwork(self,train_input, trian_table):

        nn = Classifier(
            layers=[
                Layer("Rectifier", units=80, frozen=False),
                Layer("Rectifier", units=80, frozen=False),
                Layer("Rectifier", units=80, frozen=False),
                Layer("Rectifier", units=40, frozen=False),
                Layer("Rectifier", units=40, frozen=False),
                Layer("Rectifier", units=40, frozen=False),
                Layer("Rectifier", units=20, frozen=False),
                Layer("Rectifier", units=20, frozen=False),
                Layer("Rectifier", units=20, frozen=False),
                Layer("Softmax")],
            learning_rate=0.01,
            learning_rule="sgd", batch_size=1,  # each sample is treated on its own
            loss_type="mcc", regularize="L2", weight_decay=0.00001,  # dropout_rate=0.5
            n_stable=20, f_stable=0.001,
            n_iter=100,
            # valid_set=(np.array(test_input),np.array(test_table)),#valid_size=0.1,
            # verbose=True
        )
        nn.fit(train_input, trian_table)

class Module(object):

    def save(self,clf,name):
        module_ = pickle.dumps(clf)
        with  open('/home/xgw/program/graduationDesign/design/programs'+name+'.module','w') as f:
            f.write(module_)

    def load(self,name):
        with  open('/home/xgw/program/graduationDesign/design/programs'+name+'.module','r') as f:
            module_=f.read()
        clf=pickle.loads(module_)
        return  clf