# the implementation of k means algorithm together with the classical Gaussian model learning using EM algorithm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal

data=pd.read_csv('x2.txt', sep=',', header=None).dropna(1).as_matrix()
m=data.shape[0]
n=data.shape[1]


def distance(a, b):
    return np.linalg.norm(a-b)

def kmean(k, data, dist, m, n):
    iteration=1000
    times=100
    cost=sys.maxint
    result=np.zeros([m])
    for s in range(times):
        centerlist = np.zeros([k, n])
        index = np.zeros([m])
        for i in range(k):
            index[i] = 1
        np.random.shuffle(index)
        counter = 0
        for i in range(m):
            if index[i] == 1:
                centerlist[counter] = data[i]
                counter += 1

        for _ in range(iteration):
            old = np.array(centerlist)
            for i in range(m):
                ind = 0
                dis = sys.maxint
                for j in range(k):
                    d = dist(centerlist[j], data[i])
                    if d < dis:
                        dis = d
                        ind = j
                index[i] = ind

            for i in range(k):
                counter = 0
                sum = np.zeros([n])
                for j in range(m):
                    if index[j] == i:
                        sum += data[j]
                        counter += 1
                sum /= counter
                centerlist[i] = np.array(sum)
            if np.all(old == centerlist):
                break


        c=costfunc(index,centerlist, data, dist, m)
        if c<cost:
            cost=c
            result=np.array(index)
    return result

def costfunc(index,centerlist, data, dist, m):
    rez=0
    for i in range(m):
        rez+=dist(centerlist[int(index[i])], data[i])
    return rez/m


def showrez(index, list, k, m):
    colorlist=['red', 'blue', 'yellow', 'green', 'pink', 'gray', 'black','magenta', 'aqua','gold','navy','orangered']
    y=np.zeros([m,2])
    steplist=np.zeros([k])
    counter=0
    for i in range(k):
        for j in range(m):
            if index[j]==i:
                y[counter]=list[j]
                counter+=1
        steplist[i]=counter

    for i in range(k):
        if i==0:
            px, py = y[0:int(steplist[i])].T
        else:
            px, py = y[int(steplist[i-1]):int(steplist[i])].T
        plt.scatter(px, py, color=colorlist[i])
    plt.show()


k=2
#k means clustering
index=kmean(k, data, distance, m, n)


#Gauss model learning
y=np.zeros([m, k])
pi=np.zeros([k,1])
mu=np.zeros([k,2])
cov=np.zeros([k,2,2])
e=np.ones([1,m])
for i in range(m):
    y[i][int(index[i])]=1

for i in range(100):
    #M stage
    pi=np.transpose(e.dot(y))/m
    mu = np.zeros([k, 2])
    for s in range(k):
        for j in range(m):
            mu[s]=mu[s]+y[j][s]*data[j]
    mu= mu/(pi*m)
    #print('1')
    cov = np.zeros([k, 2, 2])
    for s in range(k):
        for j in range(m):
            vec=data[j]-mu[s]
            temp=np.zeros([2,2])
            for st in range(2):
                for jt in range(2):
                    temp[st][jt]=vec[st]*vec[jt]
            cov[s]=cov[s]+y[j][s]*(temp)

    for s in range(k):
        cov[s]=cov[s]/pi[s][0]/m
#E stage
    y=np.zeros([m,k])
    for s in range(k):
        y[:,s]=multivariate_normal.pdf(data, mu[s], cov[s])*pi[s][0]
    ys=np.sum(y, axis=1)
    for j in range(m):
        y[j]=y[j]/ys[j]

print("After 100 iteration of EM algorithm, the parameter pi is:")
print(pi)
print("The average of the two cluster(mu) are:")
print(mu)
print ("The covariance matrix of the two clusters are:")
print(cov)
p = np.argmax(y, axis=1)
index=np.expand_dims(p, axis=1)
showrez(index, data, k, m)