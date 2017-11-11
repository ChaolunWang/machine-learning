#This code demonstrates using random forest to analise the madelon data set
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

train_data=pd.read_csv('madelon_train.data', sep=' ', header=None).dropna(1)
train_labels=pd.read_csv('madelon_train.labels', sep=' ', header=None)
test_data=pd.read_csv('madelon_valid.data', sep=' ', header=None).dropna(1)
test_labels=pd.read_csv('madelon_valid.labels', sep=' ', header=None)
#print('Data size: ', train_data.shape)
#print('labels size: ', train_labels.shape)
#print('Data size: ', test_data.shape)
#print('labels size: ', test_labels.shape)

#print(train_data.head())

tree_number_list=[3,10,30,100,300]
train_err_list=[]
test_err_list=[]

for i in range(5):
    tree_number=tree_number_list[i]
    #training the random forest
    random_forest=RandomForestClassifier(n_estimators=tree_number, criterion='entropy')
    random_forest.fit(train_data, train_labels.values.ravel())

    #predicting
    train_pred=random_forest.predict(train_data)
    test_pred=random_forest.predict(test_data)

    #evaluate the error
    train_error=1-accuracy_score(train_labels, train_pred)
    test_error=1-accuracy_score(test_labels, test_pred)

    print('tree numbers: ',tree_number, 'test error: ', test_error)

    #recording
    train_err_list.append(train_error)
    test_err_list.append(test_error)


plt.plot(tree_number_list, train_err_list,label='training error')
plt.plot(tree_number_list, test_err_list,label='test error')
plt.xlabel('Tree numbers')
plt.ylabel('Error rate')
plt.legend(loc=1)
plt.show()

