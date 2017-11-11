# the implementation of two steps gm algorithm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal

data=pd.read_csv('x2.txt', sep=',', header=None).dropna(1).as_matrix()
m=data.shape[0]
n=data.shape[1]


def distance(a, b):
    return np.linalg.norm(a-b)


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

def pruning(pi, mu, I): #for convinience only work for k=2
    mask=np.ones(I)
    for i in range(I):
        if pi[i][0]<1/(4*I):
            mask[i]=0
    dist=0
    x=0
    y=0
    for i in range(I):
        for j in range(I):
            if(mask[i]==1 and mask[j]==1 and distance(mu[i],mu[j])>dist):
                dist=distance(mu[i],mu[j])
                x=i
                y=j
    return (x,y)
k=2
I=10
#Gauss model learning
pi=np.zeros([I,1])
mu=np.zeros([I,2])
cov=np.zeros([I,2,2])
e=np.ones([1,m])
y=np.zeros([m,I])

for i in range(I):
    pi[i][0]=1/I
    mu[i][0]=np.random.uniform(-5,5)
    mu[i][1]=np.random.uniform(-5,5)

for i in range(I):
    mi=100000
    for j in range(I):
        if distance(mu[i], mu[j])<mi and i!=j:
            mi=distance(mu[i], mu[j])
    cov[i][0][0]=cov[i][1][1]=mi

for _ in range(50):
    #E stage
    y=np.zeros([m,I])
    for s in range(I):
        y[:,s]=multivariate_normal.pdf(data, mu[s], cov[s])*pi[s][0]
    ys=np.sum(y, axis=1)
    for j in range(m):
        y[j]=y[j]/ys[j]

    #M stage
    pi=np.transpose(e.dot(y))/m
    mu = np.zeros([I, 2])
    for s in range(I):
        for j in range(m):
            mu[s]=mu[s]+y[j][s]*data[j]
    mu= mu/(pi*m)
    #print('1')
    cov = np.zeros([I, 2, 2])
    for s in range(I):
        for j in range(m):
            vec=data[j]-mu[s]
            temp=np.zeros([2,2])
            for st in range(2):
                for jt in range(2):
                    temp[st][jt]=vec[st]*vec[jt]
            cov[s]=cov[s]+y[j][s]*(temp)

    for s in range(I):
        cov[s]=cov[s]/pi[s][0]/m

#pruing
a, b=pruning(pi, mu, I)
pi=pi[[a,b],:]
mu=mu[[a,b],:]
cov=cov[[a,b],:,:]

for _ in range(50):
    #E stage
    y=np.zeros([m,k])
    for s in range(k):
        y[:,s]=multivariate_normal.pdf(data, mu[s], cov[s])*pi[s][0]
    ys=np.sum(y, axis=1)
    for j in range(m):
        y[j]=y[j]/ys[j]

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
print("After the two step EM algorithm, the parameter pi is:")
print(pi)
print("The average of the two cluster(mu) are:")
print(mu)
print ("The covariance matrix of the two clusters are:")
print(cov)
p = np.argmax(y, axis=1)
index=np.expand_dims(p, axis=1)
showrez(index, data, k, m)