import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

def optimise_gradient_descent(X,Y,learning_rate=0.0001,n_iter=1000):
    w=np.zeros(len(X[0]))
    b=0
    t=np.where(Y<=0,-1,1)
    for i in range(1,n_iter):
        for j,x in enumerate(X):
            condition = t[j] * (np.dot(x, w) + b) >= 1
            if condition:
                w=w - learning_rate*(2*(1/i)*w)
            else:
                w = w- learning_rate * (2 * (1/i) * w - np.dot(x, t[j]))
                #b = b - learning_rate * t[j]
    return w

def get_bias(X,y,w):
    """return bias as the average of errors(true-predicted) or update it in the optimisee_gradient_descent function"""
    errors=[]
    for i in range(X.shape[0]):
        true=y[i]
        predicted=np.dot(X[i],w)
        errors.append(true-predicted)
    return sum(errors)/len(errors)

    
def visualize_svm(X,y,w,b):
    """
    Visualize the svm
    """
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])
    x1_1 = get_hyperplane_value(x0_1, w, b, 0)
    x1_2 = get_hyperplane_value(x0_2, w, b, 0)
    x1_1_m = get_hyperplane_value(x0_1, w, b, -1)
    x1_2_m = get_hyperplane_value(x0_2, w, b, -1)
    x1_1_p = get_hyperplane_value(x0_1, w, b, 1)
    x1_2_p = get_hyperplane_value(x0_2, w, b, 1)
    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'y--')
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'k')
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k')
    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min-3, x1_max+3])
    plt.title("linear svm from scratch")
    plt.tight_layout()
    plt.show()

iris = datasets.load_iris()
X = iris.data[:, :2]
#lets make it a binary classification problem by converting all 1's and 2's to 1 category(cince the iris data contains 3 target variables)
Y = (iris.target != 0)*1
xt=X
yt=Y
weights2=optimise_gradient_descent(xt,yt)
yt=np.where(Y==0,-1,1)
bias2=get_bias(xt,yt,weights2)
print(weights2,bias2)
y=np.where(Y==0,-1,1)
visualize_svm(X,y,weights2,bias2)
