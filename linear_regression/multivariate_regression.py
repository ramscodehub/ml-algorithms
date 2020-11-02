#m is noOf_Samples
#n is noOf_features
#theta is (theta0,theta1,theta2,....thetaan)
#X is feature vector where x0 is always 1
#hypothesis function h(x1,x2,x3,....xn) = theta0 + theta1*x1 + theta2*x2 +....+ thetan*xn
#h(x)=[theta0 theta1 theta2...thetan][1 x1 x2 x3 ..xn]^Transpose
#alpha is learning rate
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
#returns the hypothesis values of target Variables
def hypothesis(theta,X,n):
    h=np.ones((X.shape[0],1))
    theta = theta.reshape(1, n+1)  # [theta0 theta1 theta2...thetan]
    for i in range(0,X.shape[0]):
        h[i]=float(np.matmul(theta,X[i]))
    h=h.reshape(X.shape[0])
    return h

def gradient_Descent(theta,alpha,noOf_Iterations,h,X,y,n):
    #lets have the cost as 1 for all iterations and update it during the training process
    cost=np.ones(noOf_Iterations)
    for i in range(0,noOf_Iterations):
        theta[0]=theta[0]-((alpha/X.shape[0])*sum(h-y))
        for j in range(1,n+1):
            theta[j]=theta[j]-((alpha/X.shape[0])*sum((h-y)*(X.transpose()[j])))
        h=hypothesis(theta,X,n)
        #cost=(1/2*n)*sum((predicted-actual)^2)
        cost[i]=(1/X.shape[0])*0.5*sum(np.square(h-y))
    theta=theta.reshape(1,n+1)
    return theta,cost

def linearRegression_Multivariable(X,y,alpha,noOf_Iterations):
    n=X.shape[1]
    one_column=np.ones((X.shape[0],1))
    X=np.concatenate((one_column,X),axis=1)
    theta=np.zeros(n+1)
    h=hypothesis(theta,X,n)
    theta,cost=gradient_Descent(theta,alpha,noOf_Iterations,h,X,y,n)
    return theta,cost

def Preprocess_Data(filename):
    data=np.loadtxt(filename,delimiter=",")
    X_train = data[:]
    X_train = np.delete(X_train, 2, axis=1)
    y_train = data[:,2]
    #lets put all ones and change it while iterating
    mean = np.ones(X_train.shape[1])
    std = np.ones(X_train.shape[1])
    for i in range(0, X_train.shape[1]):
        mean[i] = np.mean(X_train.transpose()[i])
        std[i] = np.std(X_train.transpose()[i])
        for j in range(0, X_train.shape[0]):
            X_train[j][i] = (X_train[j][i] - mean[i])/std[i]
    return X_train,y_train


X_train,y_train=Preprocess_Data("data2.txt")
theta, cost = linearRegression_Multivariable(X_train, y_train,0.01,50)
print(theta,cost)
plt.plot([i for i in range(len(cost))],cost)
plt.show()
