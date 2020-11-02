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



"""data1.txt
6.1101,17.592
5.5277,9.1302
8.5186,13.662
7.0032,11.854
5.8598,6.8233
8.3829,11.886
7.4764,4.3483
8.5781,12
6.4862,6.5987
5.0546,3.8166
5.7107,3.2522
14.164,15.505
5.734,3.1551
8.4084,7.2258
5.6407,0.71618
5.3794,3.5129
6.3654,5.3048
5.1301,0.56077
6.4296,3.6518
7.0708,5.3893
6.1891,3.1386
20.27,21.767
5.4901,4.263
6.3261,5.1875
5.5649,3.0825
18.945,22.638
12.828,13.501
10.957,7.0467
13.176,14.692
22.203,24.147
5.2524,-1.22
6.5894,5.9966
9.2482,12.134
5.8918,1.8495
8.2111,6.5426
7.9334,4.5623
8.0959,4.1164
5.6063,3.3928
12.836,10.117
6.3534,5.4974
5.4069,0.55657
6.8825,3.9115
11.708,5.3854
5.7737,2.4406
7.8247,6.7318
7.0931,1.0463
5.0702,5.1337
5.8014,1.844
11.7,8.0043
5.5416,1.0179
7.5402,6.7504
5.3077,1.8396
7.4239,4.2885
7.6031,4.9981
6.3328,1.4233
6.3589,-1.4211
6.2742,2.4756
5.6397,4.6042
9.3102,3.9624
9.4536,5.4141
8.8254,5.1694
5.1793,-0.74279
21.279,17.929
14.908,12.054
18.959,17.054
7.2182,4.8852
8.2951,5.7442
10.236,7.7754
5.4994,1.0173
20.341,20.992
10.136,6.6799
7.3345,4.0259
6.0062,1.2784
7.2259,3.3411
5.0269,-2.6807
6.5479,0.29678
7.5386,3.8845
5.0365,5.7014
10.274,6.7526
5.1077,2.0576
5.7292,0.47953
5.1884,0.20421
6.3557,0.67861
9.7687,7.5435
6.5159,5.3436
8.5172,4.2415
9.1802,6.7981
6.002,0.92695
5.5204,0.152
5.0594,2.8214
5.7077,1.8451
7.6366,4.2959
5.8707,7.2029
5.3054,1.9869
8.2934,0.14454
13.394,9.0551
5.4369,0.61705"""
