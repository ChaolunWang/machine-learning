import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
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

depth_list=[]
train_err_list=[]
test_err_list=[]

for d in range(12):
    tree_depth=d+1
    #training the decition tree
    decision_tree=DecisionTreeClassifier(criterion='entropy',max_depth=tree_depth)
    decision_tree.fit(train_data, train_labels)

    #predicting
    train_pred=decision_tree.predict(train_data)
    test_pred=decision_tree.predict(test_data)

    #evaluate the error
    train_error=1-accuracy_score(train_labels, train_pred)
    test_error=1-accuracy_score(test_labels, test_pred)

    print('tree depth: ',tree_depth, 'test error: ', test_error)
    #recording
    depth_list.append(tree_depth)
    train_err_list.append(train_error)
    test_err_list.append(test_error)


plt.plot(depth_list, train_err_list,label='training error')
plt.plot(depth_list, test_err_list,label='test error')
plt.xlabel('Maximum depth')
plt.ylabel('Error rate')
plt.legend(loc=1)
plt.show()

