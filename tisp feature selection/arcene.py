#This code implement the tisp algorithm on feature selection
import numpy as np
import pandas as pd
import math
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy import stats

# reading data from the file
train_data=pd.read_csv('arcene_train.data', sep=' ', header=None).dropna(1).as_matrix()
train_labels=pd.read_csv('arcene_train.labels', sep=' ', header=None).as_matrix()
test_data=pd.read_csv('arcene_valid.data', sep=' ', header=None).dropna(1).as_matrix()
test_labels=pd.read_csv('arcene_valid.labels', sep=' ', header=None).as_matrix()
#normalize the data to mean 0 and std 1
def normalize(train, test):
    mean=np.mean(train, axis=0)
    std= np.std(train, axis=0)
    train=(train-mean)/(std+1e-7)
    test=(test-mean)/(std+1e-7)
    return train, test

train_data_norm, test_data_norm =normalize(train_data, test_data)

#take out the data size
N =  train_data_norm.shape[0] #row size
NN = train_data_norm.shape[1] #column size
TN = test_data_norm.shape[0]


#add one extra column 1s at the beginning of the data
train_data = np.hstack((np.ones((N, 1)), train_data_norm))
test_data = np.hstack((np.ones((TN, 1)), test_data_norm))

# train_labels_pro = stats.threshold(train_labels, threshmin = 1)
# test_labels_pro = stats.threshold(test_labels, threshmin = 1)

# train_labels = train_labels_pro
# test_labels = test_labels_pro

def penalty_theta(x, _lambda, _yita):
    xx = stats.threshold(x, threshmin = _lambda)
    xxx = stats.threshold(x, threshmax = - _lambda)
    x = xx + xxx
    return np.count_nonzero(x), x / ( 1 + _yita)

def iteration_steps(train_data_, train_label_, w, steps, _lambda, param,step_size):
    n=0
    N = train_data_.shape[0]
    for _ in range(steps):
        w_temp = w + np.transpose(np.transpose(train_data_).dot(train_label_- 1 / ( 1 + np.exp(-train_data_.dot(np.transpose(w))))))/N*step_size
        n,w_temp2 = penalty_theta(w_temp, _lambda, param)
        w = w_temp2
    return n,w

def linear_regression_predit(w, test_data_):
    results = []
    expvalue = np.exp(test_data_.dot(np.transpose(w)))
    p = expvalue / ( 1 + expvalue)
    for i in p:
        if i > 0.5:
            results.append(1)
        else:
            results.append(-1)
    return results

# 1, 0.2
#_lambda = 0.0054
param = 0
step_size=0.01
likehood_list=[]
xlabel=[]
train_err_list = []
test_err_list=[]
iteration=100
lambda_list=[0.0043, 0.0039,0.00359,0.00311]
para_num=0

for i in lambda_list:
    w = np.zeros(NN + 1)
    w = np.expand_dims(w, axis=0)
    para_num,w = iteration_steps(train_data, train_labels, w, iteration, i, param, step_size)
    xlabel.append(para_num)
    #predicting
    train_pred=np.asarray(linear_regression_predit(w, train_data))
    test_pred=np.asarray(linear_regression_predit(w, test_data))
    train_error=1-accuracy_score(train_labels, train_pred)
    test_error=1-accuracy_score(test_labels, test_pred)
    print('parameter number: ',para_num, 'train error: ', train_error, 'test error: ', test_error, 'lambda value: ', i)
    #recording
    train_err_list.append(train_error)
    test_err_list.append(test_error)


plt.plot(xlabel, train_err_list,label='training error')
plt.plot(xlabel, test_err_list,label='test error')
plt.xlabel('feature number')
plt.ylabel('Error rate')
plt.legend(loc=1)
plt.show()