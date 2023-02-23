# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:28:12 2021

@author: efthi
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from create_tree import Tree
import copy
from Tree_sample import TreeDistribution, forward, reverse
import random
from Metrics import stats, accuracy
import math
import numpy as np
import pandas as pd
from multiprocessing import Process, Manager

data = datasets.load_wine()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30,random_state=5)
initialTree = Tree(X_train, y_train)
a = 0.01
b = 5            
currentTree = initialTree    
forward_probs = []    
sampledTrees = []
    
forest = []
def main(currentTree, forest):

    for i in range (5000):
        state = " "
        #the below command can run outside for loop. current_tree_target is the same if rejected, otherwise is updated to new_tree_target
        forward_proposal = TreeDistribution(currentTree)
        sampleTree = forward_proposal.sample()
        forward_probability = forward_proposal.eval(sampleTree)
        forward_probability = forward(forward_probs, forward_probability)
        
        reverse_proposal = TreeDistribution(sampleTree)
        reverse_probability = reverse_proposal.eval(currentTree)
        reverse_probability = reverse(forward_probs, reverse_probability)
        
        new_tree_target = sampleTree.evaluatePosterior(a,b)
        current_tree_target = currentTree.evaluatePosterior(a,b)
        
        targetRatio = new_tree_target - current_tree_target
        proposalRatio = math.log(reverse_probability) - math.log(forward_probability)
        acceptLogProbability = min(1, math.exp(targetRatio + proposalRatio))
    
        
        q= random.random()
        if ((acceptLogProbability) > q):
            currentTree = sampleTree
            state = "accepted"

        else:
            currentTree = currentTree
            del forward_probs[-1]
            
        
        sampledTrees.append(copy.deepcopy(currentTree))
        if i >3000 and state == "accepted" :
            labels = stats.predict (currentTree, X_test)
            forest.append(labels)
    
    return forest
    


def chain(currentTree):
    with Manager() as manager: 
        L = manager.list()
        jobs = []
        for _ in range (1):#number of cores to use
            jobs.append(Process(target = main, args= [currentTree, L]))
        for i in range(len(jobs)):
            jobs[i]. start()                
        for i in range(len(jobs)):
            jobs[i]. join()
                    
        labels = list(L)
        labels = np.array(labels)
        print(labels.shape)
    return labels
    
    


        
#labels = main(currentTree, forest)
#labels = stats.predict (currentTree, X_test)

if __name__ == '__main__':

    labels = []
    forest  = chain(currentTree)
    #forest = main(currentTree, forest)    
    predictions = pd.DataFrame(forest)
    for column in predictions:
        labels.append(predictions[column].mode())
    labels =  pd.DataFrame(labels)
    labels = labels.values.tolist()
    

    acc = accuracy(y_test, labels)
    print("accuracy is: ", acc)


