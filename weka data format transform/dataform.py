# This code is for data transformation
#Add a first row containing the variable names (e.g. X1, X2, ... Y)
#Change the class labels from numeral (1,2,3,4...) to literal (e.g. C1, C2, C3...)


testdata=open('sat.tst', 'r')
trainingdata=open('sat.trn','r')

ptest=open('satw.tst', 'w')
ptrain=open('satw.trn','w')

ptest.write('X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15 X16 X17 X18 X19 X20 X21 X22 X23 X24 X25 X26 X27 X28 X29 X30 X31 X32 X33 X34 X35 X36 Y\n')
temp=''

while True:
    temp=testdata.readline()
    if temp=='':
        break
    ptest.write(temp[:-2]+'C'+temp[-2:])

ptrain.write('X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15 X16 X17 X18 X19 X20 X21 X22 X23 X24 X25 X26 X27 X28 X29 X30 X31 X32 X33 X34 X35 X36 Y\n')
while True:
    temp=trainingdata.readline()
    if temp=='':
        break
    ptrain.write(temp[:-2]+'C'+temp[-2:])

testdata.close()
trainingdata.close()
ptest.close()
ptrain.close()