# the implementation of k means algorithm
import numpy as np
import matplotlib.pyplot as plt

def distance(a, b):
    return np.linalg.norm(a-b)

def kmean(k, data, dist, m, n):
    iteration=1000
    times=100
    cost=10000000
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
                dis = 10000000
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

data=np.zeros([300,2])
for i in range(100):
    data[i*3]=np.random.normal(1,1,2)
    data[i*3+1] = np.random.normal(5, 1, 2)
    data[i * 3+2] = np.random.normal(9, 1, 2)

np.random.shuffle(data)
px, py = data[0:300].T
plt.scatter(px, py, color='red')
plt.show()

index=kmean(3, data, distance, 300, 2)
showrez(index, data, 3, 300)