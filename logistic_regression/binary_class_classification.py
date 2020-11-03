import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

def logisticFunction(z):
    return 1/(1+np.exp(-z))

def costFunction(hypothesis,y):
    return (-y*np.log(hypothesis)-(1-y)*np.log(1-hypothesis)).mean()

def logisticRegression(alpha,X,y,num_iter):
    theta=np.zeros(X.shape[1])
    for iteration in range(num_iter):
        z=np.dot(X,theta)
        hypothesis=logisticFunction(z)
        gradient=np.dot(X.T,hypothesis-y)/y.size
        theta=theta-alpha*gradient
        z=np.dot(X,theta)
        hypothesis=logisticFunction(z)
        J=costFunction(hypothesis,y)
        if iteration%100==0:
            print(f"loss: {J}")
    return theta

iris=datasets.load_iris()
X=iris.data[:,:2]
#lets make it a binary classification problem by converting all 1's and 2's to 1 category(cince the iris data contains 3 target variables)
Y=(iris.target!=0)*1
alpha=0.01
theta=logisticRegression(alpha,X,Y,num_iter=100)
print(theta)
def predictUnseenInput(X):
    return logisticFunction(np.dot(X,theta))
