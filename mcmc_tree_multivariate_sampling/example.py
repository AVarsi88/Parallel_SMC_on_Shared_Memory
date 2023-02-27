# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:28:12 2021

@author: efthi
"""

from sklearn.model_selection import train_test_split
from create_tree import Tree
from sklearn.metrics import classification_report, confusion_matrix
import copy
from Tree_sample import TreeDistribution, forward, reverse
import random
from Metrics import stats, accuracy
import math
import numpy as np
import pandas as pd
import time
from sklearn import datasets


from multiprocessing import Process, Manager
import multiprocessing

data = datasets.load_wine()

X = data.data
y = data.target
#print("data length: ", len(data))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30,random_state=5)

initialTree = Tree(X_train, y_train)


a = 0.01
b = 10


available_cores = multiprocessing.cpu_count()
print("available cores: ", available_cores)
cores_to_use = 10
iterations = 8000//cores_to_use


convergence_rate =  iterations/cores_to_use/2
currentTree = initialTree


def new_tree_target_(sampledTrees, new_trees_targets,currentTree, current_trees_targets, y_hat1):
    
    new_tree_target, _ = sampledTrees.evaluatePosterior(a,b)
    new_trees_targets.append(new_tree_target)
    current_tree_target, y_hat_var = currentTree.evaluatePosterior(a,b)
    current_trees_targets.append(current_tree_target)
    
    (y_hat1.append(y_hat_var))
    
    
    return new_trees_targets,current_trees_targets, y_hat1


def distributed_new_tree_target(sampledTrees,current_trees,cores_to_use):

    with Manager() as manager: 
            L = manager.list()
            L1 = manager.list()
            y_hat = manager.list()
            jobs = []
            for i in range (cores_to_use):#number of cores to use
                jobs.append(Process(target = new_tree_target_, args= [sampledTrees[i], L,current_trees[i], L1, y_hat]))
            for i in range(len(jobs)):
                jobs[i]. start()
            for i in range(len(jobs)):
                jobs[i]. join()
                
            y_hat_list = list(y_hat)
            labels = list(L)
            labels1 = list(L1)
         
    return labels, labels1, y_hat_list
   
forward_probs = []
sampledTrees = []
forest = []
f1 = []
store_current_trees = []   
current_trees = []
sample_from_multinomial_distribution=[cores_to_use]

for i in range(cores_to_use):
    store_current_trees.append(currentTree)
print("number of iterations: ", iterations, "and number of cores: ", cores_to_use)
if __name__ == '__main__':
    collection_for_y_hat = []
    for i in range (iterations):
        # start_time = time.time()
        print("iteration: ", i)
   
        reverse_prob= []
        forward_prob = []
        new_trees_targets = []
        current_trees_targets= []
        sampledTrees = []
        current_trees = []
  
        for p in range(len(sample_from_multinomial_distribution)):
            while sample_from_multinomial_distribution[p] !=0:
                current_trees.append(store_current_trees[p])
                sample_from_multinomial_distribution[p] -=1
        
        if  i > convergence_rate :
            for current_tree in current_trees:
                labels = stats.predict (current_tree, X_test)
                forest.append(labels)
            
            
        
        for l in range(len(current_trees)):
            forward_proposal = TreeDistribution(current_trees[l])
            sampleTree = forward_proposal.sample()
            sampledTrees.append(sampleTree)
        
        k=0
        for sample in sampledTrees:
            
            forward_probability = forward_proposal.eval(sampledTrees[k])
            reverse_proposal = TreeDistribution(sample)
            reverse_probability = reverse_proposal.eval(current_trees[k])
            forward_prob.append(forward_probability)
            reverse_prob.append(reverse_probability)
            k+=1
       
        new_tree_target, current_tree_target, y_hat = distributed_new_tree_target(sampledTrees,current_trees,cores_to_use)#this work in parallel
        
        for element in y_hat:
            collection_for_y_hat.append(element)
        targetRatio = []#substract all the new_tree_targets with the current tree
        for i in range(len(current_tree_target)):
            targetRatio.append(new_tree_target[i] - current_tree_target[i])
            
        proposalRatio = []
        for i in range(len(reverse_prob)):
            proposalRatio.append(reverse_prob[i] -forward_prob[i])
            
        store_samples_probabilities = []    
        for i in range(len(proposalRatio)):
            if targetRatio[i] + proposalRatio[i] <2:
                store_samples_probabilities.append(min(1,math.exp(targetRatio[i] + proposalRatio[i])))
            else:
                store_samples_probabilities.append(1)
        
        accepted_samples_prob = []
        store_current_trees = []
        q= random.random()
        samples_acceptance_ratio = 0
        sample_the_next_trees = []
        
        for i in range (len(store_samples_probabilities)):
            samples_acceptance_ratio += store_samples_probabilities[i]/len(store_samples_probabilities) #formila 16 overleaf
       
        if samples_acceptance_ratio > q:
            store_current_trees=copy.deepcopy(sampledTrees)
            
            for i in range(len(store_samples_probabilities)):
                accepted_samples_prob.append(store_samples_probabilities[i])
                sample_the_next_trees.append(store_current_trees[i])
        
        if len(accepted_samples_prob) == 0:#if we dont accept any sample, we sample from the highest alpha   
            max_value = max(store_samples_probabilities)
            max_index = store_samples_probabilities.index(max_value)
            accepted_samples_prob.append(store_samples_probabilities[max_index])
            for i in range(cores_to_use):
                sample_the_next_trees.append(sampledTrees[max_index])

        store_current_trees = copy.deepcopy(sample_the_next_trees)

        if (len(accepted_samples_prob) >= 1) and sum(accepted_samples_prob)>0.0001: 
              
            accepted_samples_prob = [(i)/sum(accepted_samples_prob) for i in accepted_samples_prob]#normalising each value adding up to one_formula 16 overleaf

        multinomial_distribution = np.random.multinomial(cores_to_use, accepted_samples_prob, size=1)
        print("array with weights: ", multinomial_distribution)
        sample_from_multinomial_distribution = []#convert array of arrays into list
        for items in multinomial_distribution:
            for item in items:
                sample_from_multinomial_distribution.append(item)

    
    labels = [] 
    predictions = pd.DataFrame(forest)
    print(len(predictions))
    for column in predictions:
        labels.append(predictions[column].mode())
    labels =  pd.DataFrame(labels)
    labels = labels.values.tolist()

    labels1 = []
    if len(labels[0])>1:
        for label in labels:
            labels1.append(label[0])
        acc = accuracy(y_test, labels1)
        labels = labels1
    else:        
        acc = accuracy(y_test, labels)
    print("accuracy is: ", acc)

    report = classification_report(y_test, labels)

    print(report)
    print("Confusion Matrix") 
    print( confusion_matrix(y_test, labels))   
    print(len(labels))
    data=pd.DataFrame(collection_for_y_hat)
    data.to_csv("./multivariate_big_data_big_f_space_35.csv", index=False)
    acc = accuracy(y_test, labels)
    print("accuracy: ", acc)
    
    

        
        

                
                
                
                
                
                
                
                
                
                
                
                
                
                
