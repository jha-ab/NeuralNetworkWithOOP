import numpy as np
import matplotlib.pyplot as plt


x=np.random.randint(-100,100, size=(100000,2))

w1 = np.array([[2,-1],[2,-1],[-2,-1],[-2,-1],[0,1],[0,1]])
b1 = np.array([15,1,15,1,60,50]).reshape(6,1)
v1 = (np.dot(w1, x.T) + b1)
a1 = np.sign(v1.T)

w2 = np.array([[1,0,1,0,0,0],[0,1,0,1,-1,1]])
b2 = np.array([[-1],[-1]])
v2= np.dot(w2, a1.T) + b2
a2 = np.sign(v2.T)

w3 = np.array([0,1]).reshape(1,2)
b3 = np.array([-1]).reshape(1,1)
v3 = np.dot(w3, a2.T) + b3
a3 = np.sign(v3.T) 

data=np.zeros((100000,3))
data[:,0:2]=x
data[:,2]=a3[:,0]

plt.figure()
plt.title('Output Data: Letter = A')
col= np.array(['w','w','k'])
plt.xlim(-66,66)
plt.ylim(-100,15)
plt.scatter(x[:,0],x[:,1], c=col[a3[:,0].astype(int)])
plt.show()
