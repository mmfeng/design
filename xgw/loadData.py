# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  #split
from imblearn.over_sampling import SMOTE

class LoadData(object):
    
    def __init__(self,data_num):
        print('-------------load data--------------')
        self.data=data_num
        
    def load(self):
        if self.data=='new':
            print('you will use the new data which local in  ./data/cownew1.xlsx')#../data/cownew1.xlsx            
            return loadNewData()
        elif self.data=='train':
            print('you will use the new data which local in  ./data/train.xlsx & ./data/test.xlsx')  # ../data/cownew1.xlsx
            return loadTrainData()
        elif self.data=='test':
            print('you will use the new data which local in  ./data/test.xlsx')  # ../data/cownew1.xlsx
            return loadTestData()
        elif self.data=='train_validate':
            print('you will use the new data which local in  ./data/train.xlsx & ./data/validate.xlsx')  # ../data/cownew1.xlsx
            return loadTrainvalidateData()
        elif self.data=='train_validate_selectData':
            print('you will use the new data which drop some unuseful data')  # ../data/cownew1.xlsx
            return loadTwoGroupData(2, 1,2,3, is_split=True, first='train', second='validate')
        else :
            print('data set do not exist')
            
    def preprocessing_oversample (self , data , toindex) :
#        print(data.iloc[:,3])
        x_train,x_test,y_train,y_test=train_test_split(data.iloc[:,0:toindex],data.iloc[:,6],train_size=0.8)#split data set
        x_train,x_test,y_train,y_test=np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test)
        in_oversample,out_oversample=SMOTE().fit_sample(x_train,y_train)#oversamole
        return in_oversample,out_oversample,x_test,y_test

    def preprocessing (self , data ,toindex):

        x_train,x_test,y_train,y_test=train_test_split(data.iloc[:,0:toindex],data.iloc[:,6],train_size=0.8)#split data set
        in_oversample,x_test,out_oversample,y_test=np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test)
        return in_oversample,out_oversample,x_test,y_test

    def preprocessing_test (self ,test ,isshuffle=False):
        if isshuffle==False:
            # train=train.sample(frac=1)
            x_test, y_test=test.iloc[:,0:6] ,test.iloc[:,6]
            x_test, y_test ,rf_x_test = np.array(x_test),np.array(y_test), np.array(x_test.iloc[:,0:3])
            return x_test,y_test,rf_x_test
        else :
            test=test.sample(frac=1)
            x_test, y_test = test.iloc[:,0:6] ,test.iloc[:,6]
            x_test, y_test, rf_x_test = \
                np.array(x_test), np.array(y_test), np.array(x_test.iloc[:,0:3])
            return x_test, y_test, rf_x_test

    def preprocessing_train_test (self ,train ,test  ,isshuffle=False):
        if isshuffle==False:
            train=train.sample(frac=1)#only shuffle training data
            x_train,x_test,y_train,y_test=train.iloc[:,0:6] ,test.iloc[:,0:6] ,train.iloc[:,6] ,test.iloc[:,6]
            in_oversample ,x_test,out_oversample ,y_test , rf_in_oversample ,rf_x_test = \
                np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test),np.array(x_train.iloc[:,0:3]),np.array(x_test.iloc[:,0:3])
            return in_oversample,out_oversample,x_test,y_test, rf_in_oversample ,rf_x_test
        else :
            data=pd.concat([train ,test], ignore_index=True)
            train=train.sample(frac=1)
            x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 0:6], data.iloc[:, 6],
                                                                train_size=0.8)
            in_oversample, x_test, out_oversample, y_test, rf_in_oversample, rf_x_test = \
                np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test),np.array(x_train.iloc[:,0:3]),np.array(x_test.iloc[:,0:3])
            return in_oversample,out_oversample,x_test,y_test, rf_in_oversample ,rf_x_test
 
def loadNewData():
    run=pd.read_excel('/home/xgw/program/graduationDesign/design/data/cownew1.xlsx',sheet_name='run',usecols=[1,2,3],header=None)
    walk=pd.read_excel('/home/xgw/program/graduationDesign/design/data/cownew1.xlsx',sheet_name='walk',usecols=[1,2,3],header=None)
    stand=pd.read_excel('/home/xgw/program/graduationDesign/design/data/cownew1.xlsx',sheet_name='stand',usecols=[1,2,3],header=None)
    standing=pd.read_excel('/home/xgw/program/graduationDesign/design/data/cownew1.xlsx',sheet_name='standing',usecols=[1,2,3],header=None).dropna(axis=0)
    lie=pd.read_excel('/home/xgw/program/graduationDesign/design/data/cownew1.xlsx',sheet_name='lie',usecols=[1,2,3],header=None)
    lying=pd.read_excel('/home/xgw/program/graduationDesign/design/data/cownew1.xlsx',sheet_name='lying',usecols=[1,2,3],header=None).dropna(axis=0)
    
    run[6]=0
    walk[6]=1
    lie[6]=2
    stand[6]=3
    standing[6]=4
    lying[6]=5
    #data=eat.append(walk).sort_values('id')
#    data=pd.concat([walk,run], ignore_index=True)
#    data=pd.concat([walk,lie,run,stand], ignore_index=True)
    data=pd.concat([walk,lie,run,stand,standing,lying], ignore_index=True)
#    data=pd.concat([standing,lying], ignore_index=True)

    
    #    三轴角度
    #    加速传感器x轴与自然坐标系x轴夹角：∠1=arctan(x/sqrt(y**2+z**2))
    
    data[4]=np.arctan(data[0]/np.sqrt(data[1]**2+data[2]**2))

    #    加速传感器y轴与自然坐标系y轴夹角：∠1=arctan(y/sqrt(x**2+z**2))
    data[3]=np.arctan(data[1]/np.sqrt(data[0]**2+data[2]**2))

    #    加速传感器z轴与自然坐标系z轴夹角：∠1=arctan(sqrt(x**2+y**2)/z)  
    data[5]=np.arctan(np.sqrt(data[0]**2+data[1]**2)/data[2])



#    x,y,z,color=data[0],data[1],data[2],data[3]

#        r=x
#        c=y
#        f=z
    
#        xyzcw=pd.concat([r,c,f,color,data.iloc[:,4:7]],axis=1)
    
#        return r,c,f,color,xyzcw
    return data.reindex(index=None,columns=[0,1,2,3,4,5,6])

def loadTwoGroupData(num,*args,is_split=False,**kwargs):
    if num==1:
        test = pd.read_excel('/home/xgw/program/graduationDesign/design/data/'+kwargs['first']+'.xlsx'
                             , header=None)
        #    三轴角度
        #    加速传感器x轴与自然坐标系x轴夹角：∠1=arctan(x/sqrt(y**2+z**2))

        test[5] = np.arctan(test[0] / np.sqrt(test[1] ** 2 + test[2] ** 2))

        #    加速传感器y轴与自然坐标系y轴夹角：∠1=arctan(y/sqrt(x**2+z**2))
        test[4] = np.arctan(test[1] / np.sqrt(test[0] ** 2 + test[2] ** 2))

        #    加速传感器z轴与自然坐标系z轴夹角：∠1=arctan(sqrt(x**2+y**2)/z)
        test[6] = np.arctan(np.sqrt(test[0] ** 2 + test[1] ** 2) / test[2])

        if is_split == True:
            test_split=pd.DataFrame([d for d in test.values if d[3] not in args])
            return test.reindex(index=None, columns=[0, 1, 2, 4, 5, 6, 3]),test_split.reindex(index=None, columns=[0, 1, 2, 4, 5, 6, 3])
        else:
            return test.reindex(index=None, columns=[0, 1, 2, 4, 5, 6, 3])

    elif num==2:
        data = pd.read_excel('/home/xgw/program/graduationDesign/design/data/'+kwargs['first']+'.xlsx'
                             , header=None)
        test = pd.read_excel('/home/xgw/program/graduationDesign/design/data/'+kwargs['second']+'.xlsx'
                             , header=None)
        #    三轴角度
        #    加速传感器x轴与自然坐标系x轴夹角：∠1=arctan(x/sqrt(y**2+z**2))

        data[5] = np.arctan(data[0] / np.sqrt(data[1] ** 2 + data[2] ** 2))
        test[5] = np.arctan(test[0] / np.sqrt(test[1] ** 2 + test[2] ** 2))

        #    加速传感器y轴与自然坐标系y轴夹角：∠1=arctan(y/sqrt(x**2+z**2))
        data[4] = np.arctan(data[1] / np.sqrt(data[0] ** 2 + data[2] ** 2))
        test[4] = np.arctan(test[1] / np.sqrt(test[0] ** 2 + test[2] ** 2))

        #    加速传感器z轴与自然坐标系z轴夹角：∠1=arctan(sqrt(x**2+y**2)/z)
        data[6] = np.arctan(np.sqrt(data[0] ** 2 + data[1] ** 2) / data[2])
        test[6] = np.arctan(np.sqrt(test[0] ** 2 + test[1] ** 2) / test[2])

        if is_split == True:
            data_split=pd.DataFrame([d for d in data.values if d[3] not in args])
            test_split=pd.DataFrame([d for d in test.values if d[3] not in args])
            return data.reindex(index=None, columns=[0, 1, 2, 4, 5, 6, 3]), \
                   test.reindex(index=None,columns=[0, 1, 2, 4, 5, 6, 3]), \
                   data_split.reindex(index=None, columns=[0, 1, 2, 4, 5, 6, 3]), \
                   test_split.reindex(index=None,columns=[0, 1, 2, 4, 5, 6, 3]),
        else:
            return data.reindex(index=None, columns=[0, 1, 2, 4, 5, 6, 3]), test.reindex(index=None,
                                                                                     columns=[0, 1, 2, 4, 5, 6, 3])


def loadTrainData():
    return loadTwoGroupData(2,first='train',second='test')

def loadTrainvalidateData():
    return loadTwoGroupData(2,first='train',second='validate')

def loadTestData():
    return loadTwoGroupData(1, first='test')

# def loadSelectTrainData(*drop_args):
#     # print(drop_args)
#


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((6, 1))
    e[j] = 1.0
    return e
