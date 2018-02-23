#!/usr/bin/env python
'''
Decision Tree Implementation

Assumption: It assumes that the last column of the dataset is class labels

Command to execute: python filemane.py gini ----> for splitting based on gini
                    python filemane.py gain ----> for splitting based in gain
'''
import pandas as pd
import random
import math
import sys
from collections import Counter
import numpy as np
# import matplotlib.pyplot as plt 

#get all unique values in columns
def unique(data, col):
	return set(i[col] for i in data)

#returns a dictionary of class labels with total counts
def get_labels(data):
	labels = Counter([i[-1] for i in data])
	return labels

#make splits based on a selected attribute and threshold value:
#return two splits, one lower than threshold and other higher 
def get_split(col, value, data):
	low = [i for i in data if i[col] < value]
	high = [i for i in data if i[col] >= value]
	return low, high

#calculates gini of a node
def gini(data):
	labels = get_labels(data)
	gini = 0
	for i in labels:
		prop = (labels[i]/float(len(data)))**2 
		gini += prop
	return 1 - gini

#calculates entropy of a node
def entropy(data):
	labels = get_labels(data)
	e = 0
	for i in labels:
		prop = labels[i]/float(len(data))
		e += -(prop * math.log(prop, 2))
	return e

#iterates over all values of a attribute and finds the one with
#maximum information gain of minimum gini based on which criteria is selected
#returns a node information stored in a dictionary with attribute to split on,
#threshold value, and lower, higher splits
def eval_split(data):
	best_i, best_val, best_gini, best_gain, best_split = 1000, 1000, 1000, 0, None 
	for i in range(len(data[0])-1):
		unique_vals = unique(data, i)
		for val in unique_vals:
			splits = get_split(i, val, data)
			p = float(len(splits[0])) / (len(splits[0]) + len(splits[1]))
			if method == 'gini':
				gini_split = p * gini(splits[0]) + (1 - p) * gini(splits[1])
				if gini_split < best_gini:
					best_i, best_val, best_gini, best_split = i, val, gini_split, splits
			if method == 'gain':
				gain_split = entropy(data) - (p * entropy(splits[0]) + (1 - p) * entropy(splits[1]))
				if gain_split > best_gain:
					best_i, best_val, best_gain, best_split = i, val, gain_split, splits
	return {'i': best_i, 'val': best_val, 'splits': best_split}

#defines leaf node and returns the class label based on majority occurances
def leaf(data):
	labels = get_labels(data)
	return max(labels, key=labels.get)

#recursive splitting until all observations are of same class or the node has 
#observations less than 20% of the training data
def split(node):
	lower, higher = node['splits'][0], node['splits'][1]
	labels_lower, lables_higher = get_labels(lower), get_labels(higher)
	if len(lower) == 0 or len(higher) == 0:
		node['low'] = node['high'] = leaf(lower + higher)
		return
	if len(labels_lower) == 1 or len(lower) < half_val * 0.2:
		node['low'] = leaf(lower)
	else:
		node['low'] = eval_split(lower)
		split(node['low'])
	if len(lables_higher) == 1 or len(higher) < half_val * 0.2:
		node['high'] = leaf(higher)
	else:
		node['high'] = eval_split(higher)
		split(node['high'])

#build tree
def tree(data):
	root_node = eval_split(data)
	split(root_node)
	return root_node
	
#making prediction
def classify(node, data):
	lower = classify(node['low'], data) if isinstance(node['low'], dict) else node['low']
	higher = classify(node['high'], data) if isinstance(node['high'], dict) else node['high']
	return lower if data[node['i']] < node['val'] else higher

def main(data):
	indices = random.sample(range(len(data)), half_val)
	data_train = [data[i] for i in indices]
	data_test = [data[i] for i in range(len(data)) if i not in indices]
	test_labels = [i[-1] for i in data_test]
	mytree = tree(data_train)
	predict_train = [classify(mytree, i) for i in data_train]
	predict_test = [classify(mytree, i) for i in data_test]
	corr_train, corr_test = 0, 0
	for i in range(len(predict_test)):
		if predict_test[i] == test_labels[i]:
			corr_test += 1
	for i in range(len(predict_train)):
		if predict_train[i] == data_train[i][-1]:
			corr_train += 1
	train_acc = float(corr_train)/len(predict_train)*100
	test_acc = float(corr_test)/len(predict_test)*100
	return train_acc, test_acc

if __name__ == "__main__":
	method = sys.argv[1]
	#link of iris dataset
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
	
	#link of wisconsin breast cancer dataset; uncomment the line below
	#to run on wisconsin breast cancer dataset
	# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
	
	#link of ionosphere dataset;  uncomment the line below
	#to run on ionosphere dataset
	# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
	
	dataset = pd.read_csv(url, sep=',')
	#removes ? and ids from wisconsin breast canser dataset
	# dataset = dataset.iloc[:, 1:]
	# dataset = dataset[dataset.iloc[:,5] != '?']
	dataset = dataset.values.tolist()
	half_val = len(dataset)/2
	train_accuracy, test_accuracy = [], []
	for i in range(5):
		train_acc, test_acc = main(dataset)
		train_accuracy.append(train_acc)
		test_accuracy.append(test_acc)
	print "train: ", train_accuracy, "test: ", test_accuracy,\
	"avg_train: ", np.mean(train_accuracy), "avg_test: ", np.mean(test_accuracy)
	# x = [1, 2, 3, 4, 5]
	# plt.scatter(x, train_accuracy)
	# plt.scatter(x, test_accuracy)
	# plt.show()

