import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def partition(data,column,value):
    """partition the feature space into left and right subtrees"""
    left=data[data[column]<=value].index
    right=data[data[column]>value].index
    return left,right

def gene_impurity(label,label_idx):
    """minimise the gene cost function;provides an indication of how pure the nodes are
    Inputs:
    1.label:class label available at current node
    2.label indices of the labels
    Output:
    returns the gene impurity of node
    """
    unique_label,unique_label_count=np.unique(label.loc[label_idx],return_counts=True)
    impurity=0
    for i in range(len(unique_label)):
        p_i=unique_label_count[i]/sum(unique_label_count)
        impurity +=p_i*(1-p_i)
    return impurity

def gene_gain(label,left_idx,right_idx,impurity):
    """Inputs:
    left node indices of a tree
    right node indices of a tree
    gene impurity of parent node
    Output:
    returns the information gain at the node"""
    p=len(left_idx)/(len(left_idx)+len(right_idx))
    info_gain=impurity-((p*gene_impurity(label,left_idx))+((1-p)*gene_impurity(label,right_idx)))
    return info_gain

def find_best_split(df,label,idx):
    """Inputs:
    df:the training data
    label:the target label
    idx:the index of the data
    Outputs:
    1.best gene gain
    2.best column: the column that produced the max gene gain
    3.the value of the column that produced the max gene gain"""
    best_gain=0
    best_col=None
    best_val=None
    #convert training data to pandas dataframe
    df=df.loc[idx]
    #getting the indices of labels of data
    label_idx=label.loc[idx].index
    #get the impurity at the current node
    impurity=gene_impurity(label,label_idx)
    #iterate through the columns and store the unique values in a set to find the best split point
    for col in df.columns:
        unique_values=set(df[col])
        #will loop through each unique value and split the data to left and right nodes
        for val in unique_values:
            left_idx,right_idx=partition(df,col,val)
            #if no condition met
            if len(left_idx)==0 or len(right_idx)==0:
                continue
            #determine the gene gain at each node
            infoGain=gene_gain(label,left_idx,right_idx,impurity)
            #if info gain greater than current gain the then that becomes the best gain
            if infoGain>best_gain:
                best_gain,best_col,best_val=infoGain,col,val
    return best_gain,best_col,best_val

def count(label,idx):
    """Counts the unique values
    Inputs:
    1.target labels
    2.index of rows
    Outputs:
    dictionary of labels and counts"""
    unique_label, unique_label_count = np.unique(label.loc[idx], return_counts=True)
    dic_label_count=dict(zip(unique_label,unique_label_count))
    return dic_label_count

class Leaf_Node:
    """A Leaf node classifies data into one of the target classes """
    def __init__(self,label,idx):
        self.predictions=count(label,idx)

class Decision_Node:
    """ Decision Node asks a question and partition the data 
    This holds a reference to the question and the two child nodes"""
    def __init__(self,column,value,true_branch,false_branch):
        self.column=column
        self.value=value
        self.true_branch=true_branch
        self.false_branch=false_branch

def build_decision_tree(df,label,idx):
    """Recursively builds the decision tree until leaf nodes are pure
    Inputs:
    1.training data
    2.target labels
    3.indexex
    Outputs:
    1.best column
    2.best value
    3.true branch
    4.false branch"""
    best_gain,best_col,best_val=find_best_split(df,label,idx)
    if best_gain==0:
        return Leaf_Node(label,label.loc[idx].index)
    left_idx,right_idx=partition(df.loc[idx],best_col,best_val)
    truebranch=build_decision_tree(df,label,left_idx)
    falsebranch=build_decision_tree(df,label,right_idx)
    return Decision_Node(best_col,best_val,truebranch,falsebranch)

def print_decision_tree(node,spacing=""):
    """Input:
    1.Tree node
    2.spacing used to space tree like data structure """
    #base case
    if isinstance(node,Leaf_Node):
        print(spacing +  "predict", node.predictions)
        return
    print(spacing + f"[{node.column} <= {node.value}]")
    #recursion on the true branch
    print(spacing + "-->True:")
    print_decision_tree(node.true_branch, spacing="  ")
    #recursion on the false branch
    print(spacing + "-->False")
    print_decision_tree(node.false_branch,spacing="  ")

def predictions(test_data,tree):
    """
    predicts the unseen examples
    Inputs:
    1.test_data
    2.trained decison tree
    Output:
    The class to which it belongs.
    """
    #if we are at leaf node
    if isinstance(tree,Leaf_Node):
        return max(tree.predictions)
    feature_name,feature_val=tree.column,tree.value
    if test_data[feature_name]<=feature_val:
        return predictions(test_data,tree.true_branch)
    else:
        return predictions(test_data,tree.false_branch)

    
#loading the data into pandas Dataframe
dataset = load_iris()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
target_zip = dict(zip(set(dataset.target), dataset.target_names))
df["target"]=dataset.target
df["target_names"]=df["target"].map(target_zip)
X=df.iloc[:,:4]
y=df.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,shuffle=True,random_state=1)
mytree=build_decision_tree(X_train,y_train,X_train.index)
X_test["predictions"]=X_test.apply(predictions,axis=1,args=(mytree,))
print(accuracy_score((y_test.loc[y_test.index]),X_test["predictions"]))
