#The implementation of Multiclass classification follows the same ideas as the binary classification.
# As we know in binary classification, we replace two classes with 1 and 0 respectively.
# In one vs all method, when we work with a class, that class is denoted by 1 and the rest of the classes becomes 0.
#we will implement logistic regression for each class. There will be a series of theta for each class as well.
#and we pass the pass the unseen example it does predict the output with all learnt parameters and outputs the most probability one



from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#loading data
iris=datasets.load_iris()
X=iris.data
y=iris.target
#standardise
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
#split into training and testing
xtrain=iris.data[:len(X)-5]
ytrain=iris.target[:len(y)-5]
xtest=iris.data[-5:]
ytest=iris.target[-5:]
# Create one-vs-rest logistic regression object
oneVsRest=LogisticRegression(multi_class="ovr")
model=oneVsRest.fit(xtrain,ytrain)
ypredicted=model.predict(xtest)
print(ypredicted)
#prints the probabilities for each model(models equal to number of classes)
print(model.predict_proba(xtest))


