import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from svm_basic import svm_basic

data = pd.read_csv('linearly_separable.csv',delimiter=",",header=None)
Xin = data.iloc[:,:2]
Yin = data.iloc[:,2]
dataArr = Xin.values.tolist()
labelArr = Yin.values.tolist()
splitpoint = int(len(labelArr)*0.5)
train_x = dataArr[:splitpoint]
train_y = labelArr[:splitpoint]
test_x = dataArr[splitpoint:]
test_y = labelArr[splitpoint:]

svm = svm_basic(1, 0.001, 50)
svm.fit(train_x,train_y)

w0 = svm.weights.item(0)
w1 = svm.weights.item(1)
b = svm.b.item(0)

print("parameters: ")
print(w0,w1,b)

svm.predict(train_x,train_y)
svm.predict(test_x,train_y)

n = Xin.shape[0]
xcord1 = []
ycord1 = []
xcord2 = []
ycord2 = []
for i in range(n):
    if int(Yin.iloc[i]) == 1:
        xcord1.append(Xin.iloc[i][0])
        ycord1.append(Xin.iloc[i][1])
    else:
        xcord2.append(Xin.iloc[i][0])
        ycord2.append(Xin.iloc[i][1])
fig = plt.figure()
    
#Plot the data as points with different colours
ax = fig.add_subplot(111)
ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
ax.scatter(xcord2, ycord2, s=30, c='green')

# Plot the best-fit line
x = np.arange(-2.0, 12.0, 0.1)
y = (-w0*x - b)/w1
ax.plot(x,y)
ax.axis([-2,8,-2,8])
plt.show()
plt.close()