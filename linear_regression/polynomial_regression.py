#polynomial regression is a form of regression analysis in which the relationship between independent variable x and dependent variable y is modelled as an nth degree polynomial in x
#polynomial regression fits a non linear relationship between fetures and a target variable.
#hypothesis:theta0 + theta1*x1 +theta2*(x1^2) + .....thetan**(xn^n)
#[theta0 theta1 theta2 theta3.....thetan]*[1 x x^2 x^3....x^n].transpose()
#cost function same as multivariate regression
import numpy as np
import matplotlib.pyplot as plt
def hypothesis(theta,X,degree):
    h=np.ones((X.shape[0],1))
    theta=theta.reshape(1,degree+1)
    for i in range(0,X.shape[0]):
        x_array=np.ones(degree+1)
        for j in range(0,degree+1):
            x_array[j]=pow(X[i],j)
        x_array=x_array.reshape(degree+1,1)
        h[i]=float(np.matmul(theta,x_array))
    h=h.reshape(X.shape[0])
    return h

def gradientDescent(theta,alpha,num_iter,h,X,y,degree):
    cost=np.ones(num_iter)
    for i in range(0,num_iter):
        theta[0]=theta[0]-((alpha/X.shape[0])*sum(h-y))
        for j in range(1,degree+1):
            theta[j]=theta[j]-((alpha/X.shape[0])*sum((h-y)*pow(X,j)))
        h=hypothesis(theta,X,degree)
        cost[i]=(1/X.shape[0])*0.5*sum(np.square(h-y))
    theta=theta.reshape(1,degree+1)
    return theta,cost


def polynomial_regression(X,y,alpha,num_iter,degree):
    theta=np.zeros(degree+1)
    h=hypothesis(theta,X,degree)
    theta,cost=gradientDescent(theta,alpha,num_iter,h,X,y,degree)
    return theta,cost


data = np.loadtxt("data1.txt",delimiter=",")
X_train = data[:, 0]
y_train = data[:, 1]
theta,cost=polynomial_regression(X_train,y_train,0.00001,100,2)
plt.plot([i for i in range(len(cost))],cost)
plt.show()
