#This code demonstrates using random forest to analise the madelon data set
import numpy as np
import pandas as pd
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import preprocessing

# reading data from the file
train_data=pd.read_csv('gisette_train.data', sep=' ', header=None).dropna(1).as_matrix()
train_labels=pd.read_csv('gisette_train.labels', sep=' ', header=None).as_matrix()
test_data=pd.read_csv('gisette_valid.data', sep=' ', header=None).dropna(1).as_matrix()
test_labels=pd.read_csv('gisette_valid.labels', sep=' ', header=None).as_matrix()
#normalize the data to mean 0 and std 1
train_data_norm = preprocessing.scale(train_data)
test_data_norm = preprocessing.scale(test_data)
#take out the data size
N =  train_data_norm.shape[0] #row size
NN = train_data_norm.shape[1] #column size
TN = test_data_norm.shape[0]
#add one extra column 1s at the beginning of the data
train_data = np.hstack((np.ones((N, 1)), train_data_norm))
test_data = np.hstack((np.ones((TN, 1)), test_data_norm))

#functioin for the partial derivative
def partial_derivative(train_data_, train_label_, index, w):
    N = train_data_.shape[0]
    derivative = 0.0
    for i in range(N):
        cofficiency = -train_data_[i][index] * train_label_[i]
        exp_number = -train_label_[i] * np.inner(train_data_[i,:], w)
        derivative = derivative +  cofficiency * math.exp(exp_number) / ( 1 + math.exp(exp_number))
    return 1./N * derivative

def gradient_descent(train_data_, train_label_, w, steps, _lambda, param):
    N = train_data_.shape[0]
    dimension = train_data_.shape[1] #dimension of the variable
    w_temp = np.zeros(dimension)
    for s in range(steps):
        for i in range(dimension):
            print (i, "in the ",  dimension)
            w_temp[i] = w[i] - param * _lambda * w[i] + param / N * partial_derivative(train_data_, train_label_, i, w) + _lambda * w[i]
        w = w_temp
    return w

def log_likehood(train_data_, train_label_, w):
    N = train_data_.shape[0]
    likeihood = 0.0
    for i in range(N):
        likeihood = likeihood + math.log(1 + math.exp(train_label_[i] * np.inner(train_data_[i,:], w)))
    return likeihood

def linear_regression_predit(w, test_data_):
    N = test_data_.shape[0]
    results = []
    for i in range(N):
        expvalue = math.exp(np.inner(test_data_[i, :],w))
        p = expvalue / ( 1 + expvalue)
        print(p)
        if p > 0.5:
            results.append(1)
        else:
            results.append(-1)
    return results

_lambda = 0.001
param = 1e-5
likehood_list=[]
likehood_xlabel=[]
train_err_list = []
test_err_list=[]
w = np.zeros(NN + 1)
iteration_list = [1,  1, 1, 1, 1]
steps_ = 0
for i in iteration_list: #range(300, 1001):
    #for j in range(i):
    if i == iteration_list[0]:
        steps_ = iteration_list[0]
    else:
        steps_ = steps_ + i
    #print "starting gradient descent"
    w = gradient_descent(train_data, train_labels, w, i, _lambda, param)
    #print "gradient descent finished"
    likehood = log_likehood(train_data, train_labels, w)
    likehood_list.append(likehood)
    likehood_xlabel.append(steps_)
    #predicting
    train_pred=np.asarray(linear_regression_predit(w, train_data))
    test_pred=np.asarray(linear_regression_predit(w, test_data))
    # print test_data.shape
    # print test_pred.shape
    # print test_labels.shape
    #evaluate the error
    train_error=1-accuracy_score(train_labels, train_pred)
    test_error=1-accuracy_score(test_labels, test_pred)
    print('iteration number: ',i, 'test error: ', test_error)
    #recording
    train_err_list.append(train_error)
    test_err_list.append(test_error)

plt.plot(likehood_xlabel, train_err_list,label='training error')
plt.plot(likehood_xlabel, test_err_list,label='test error')
plt.plot(likehood_xlabel, likehood_list,label='likeihood')
plt.xlabel('iteration steps')
plt.ylabel('Error rate')
plt.legend(loc=1)
plt.show()
plt.plot(likehood_xlabel, likehood_list,label='likeihood')
plt.xlabel('iteration steps')
plt.ylabel('likeihood')
plt.legend(loc=1)
plt.show()
