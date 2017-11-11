#This code demonstrates using random forest to analise the madelon data set
import numpy as np
import pandas as pd
import math
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy import stats
import heapq

# reading data from the file
train_data=pd.read_csv('madelon_train.data', sep=' ', header=None).dropna(1).as_matrix()
train_labels=pd.read_csv('madelon_train.labels', sep=' ', header=None).as_matrix()
test_data=pd.read_csv('madelon_valid.data', sep=' ', header=None).dropna(1).as_matrix()
test_labels=pd.read_csv('madelon_valid.labels', sep=' ', header=None).as_matrix()
#normalize the data to mean 0 and std 1
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

def penalty_theta(x,mu, k, i, n):
    M=x.shape[1]
    m=k+(M-k)*max(0,(n-2*i)/(2*i*mu+n))
    index=heapq.nlargest(int(m), range(M), np.absolute(x).take)
    temp=np.zeros(M)
    for i in index:
        temp[i]=x[0][i]
    temp = np.expand_dims(temp, axis=0)
    return temp


def iteration_steps(train_data_, train_label_, w, steps, mu, k ,step_size):
    N = train_data_.shape[0]
    steplist=[]
    lostlist=[]
    for i in range(steps):
        w_temp = w + 2*np.transpose(train_label_-train_data_.dot(np.transpose(w))).dot(train_data_)*step_size/N-2*0.001*step_size*w
        w_temp2 = penalty_theta(w_temp, mu, k, i, steps)
        w = w_temp2
        steplist.append(i+1)
        lostlist.append(np.linalg.norm(train_label_-train_data_.dot(np.transpose(w)))**2+0.001*np.linalg.norm(w)**2)

    if(k==10):
        plt.plot(steplist, lostlist)
        plt.xlabel('Iterations')
        plt.ylabel('Regression loss')
        plt.legend(loc=1)
        plt.show()

    return w

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
step_size=0.001
xlabel=[]
train_err_list = []
test_err_list=[]
iteration=500
k_list=[10, 30, 100, 300]
mu=100

for i in k_list:
    w = np.zeros(NN + 1)
    w = np.expand_dims(w, axis=0)
    w = iteration_steps(train_data, train_labels, w, iteration, mu, i , step_size)
    xlabel.append(i)
    #predicting
    train_pred=np.asarray(linear_regression_predit(w, train_data))
    test_pred=np.asarray(linear_regression_predit(w, test_data))
    train_error=1-accuracy_score(train_labels, train_pred)
    test_error=1-accuracy_score(test_labels, test_pred)
    print('parameter number: ',i, 'train error: ', train_error, 'test error: ', test_error)
    #recording
    train_err_list.append(train_error)
    test_err_list.append(test_error)


plt.plot(xlabel, train_err_list,label='training error')
plt.plot(xlabel, test_err_list,label='test error')
plt.xlabel('feature number')
plt.ylabel('Error rate')
plt.legend(loc=1)
plt.show()
