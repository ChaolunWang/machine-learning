# the python version of evolutionary algorithm

import math
import numpy as np

def costFunc(list, n):
	return -1*np.linalg.norm(list[0:int(n/2)])+np.linalg.norm(list[int(n/2):])

def rng(min, max, value):
	return np.floor((value-min)/(max-min)+np.random.rand())

def crossing(arr1, arr2, n):
	k=int(np.floor(np.random.rand()*n))
	temp=list(arr1[k:])
	arr1=arr1[:k]
	temp2=list(arr2[k:])
	arr1.extend(temp2)
	arr2=arr2[:k]
	arr2.extend(temp[:])
	
def reverse(i, r):
	if(np.floor(np.random.rand()+r)):
		if i==0:
			return 1
		else:
			return 0
	return i
	
def mutate(list, mutationRate):
	return [reverse(i, mutationRate) for i in list]
	
def reproduce(genePool, index, compete, n, mutationRate):
	mates=np.where(compete==compete.max())
	mate=mates[0][0]
	#print('mate='+str(mate))
	arr1=mutate(genePool[index, :], mutationRate)
	arr2=mutate(genePool[mate, :], mutationRate)
	#print(arr1)
	#print(genePool)
	#print(compete)
	crossing(arr1, arr2, n)
	#print(arr1)
	genePool=np.append(genePool, [arr1], axis=0)
	genePool=np.append(genePool, [arr2], axis=0)
	#print(genePool)
	#print(costFunc(arr1, n))
	compete=np.append(compete, costFunc(arr1, n))
	compete=np.append(compete, costFunc(arr2, n))
	
	for i in range(2):
		tempt=np.where(compete==compete.min())
		temp=tempt[0][0]
		#temp=(compete.tolist()).index(max(compete.tolist()))
		genePool=np.delete(genePool, temp, 0)
		compete=np.delete(compete, temp, 0)
		
	print(np.sum(compete)/m)
	return (genePool, compete)
	
def evolve(genePool, mutationRate, min, max, m, n, iteration):
	compete=np.array([costFunc(genePool[i], n) for i in range(m)])
	for i in range(iteration):
		index=int(np.floor(np.random.rand()*m))
		if rng(min, max, compete[index]):#True:
			(genePool,compete)=reproduce(genePool, index, compete, n, mutationRate)
	return genePool

genePool=np.random.rand(10, 5)
genePool=np.floor(genePool+0.5)
print(genePool)
np.random.shuffle(genePool)
print(genePool)
print(costFunc(genePool[0,:], 5))
print(-1*np.linalg.norm([1]*5))

m=100
n=1000
min=-1*math.sqrt(n/2)
max=math.sqrt(n/2)
genePool=np.random.rand(m, n)
genePool=np.floor(genePool+0.5)
genePool=evolve(genePool, 0.05, min, max, m, n, 1000)
print(genePool)
print('min='+str(min)+'max='+str(max))
