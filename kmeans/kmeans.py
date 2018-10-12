#!/usr/bin/env python

'''
Kmeans Implementation

Assumption: It assumes that the last column of the dataset is class labels

Command to execute: python filemane.py euclidean ----> for euclidean distance
                    python filemane.py cosine ----> for cosine distance
                    python filemane.py citiblock ----> for manhattan distance
'''

import pandas as pd
import random
import numpy as np
import sys
from collections import Counter

def euclidean(a, b):
    return np.sqrt(np.sum((b - a)**2, axis=1))

def cosine_dist(a, b):
	a_l2norm = np.sum(a**2)**(1./2)
	b_l2norm = np.sum(b**2, axis=1)**(1./2)
	return 1 - (np.dot(a, b.T)/(a_l2norm * b_l2norm))

def citiblock(a, b):
	return np.sum(np.absolute(b-a), axis=1)

def dist3(a, b):
	diff = a - b
	dist = ((np.sum((diff).clip(0), axis=1))**2 + \
		(np.sum((-diff).clip(0), axis=1))**2)**(0.5)
	return dist
	
def dist4(a, b):
	diff = a - b
	dist = ((np.sum((diff).clip(0), axis=1))**2 + \
		(np.sum((-diff).clip(0), axis=1))**2)**(0.5)
	a = np.array([a]*len(b))
	max_array =  np.sum(np.maximum.reduce([abs(a), abs(b), abs(diff)], axis=0), axis=1)
	final_dist = dist.astype('float64') / max_array.astype('float64')
	return final_dist
	
def get_distance(centroid):
	dist_measure = {i: [] for i in range(len(main_data))}
	for i,j in centroid.items():
		if method == 'euclidean':
			dist_i = euclidean(j, data)
		if method == 'cosine':
			dist_i = cosine_dist(j, data)
		if method == 'citiblock':
			dist_i = citiblock(j, data)
		if method == 'dist3':
			dist_i = dist3(j, data)
		if method == 'dist4':
			dist_i = dist4(j, data)
		for p in range(len(dist_i)):
			dist_measure[p].append([i, dist_i[p]])
	return dist_measure
		
def allocate_cluster(centroid):
	dist_data = get_distance(centroid)
	for i in dist_data:
		dist_data[i] = sorted(dist_data[i], key=lambda x : x[1])
	clusters = {i: dist_data[i][0][0] for i in dist_data}
	return clusters

def get_new_centroid(centroid):
	old_cluster = allocate_cluster(centroid)
	assigned_clusters = set(old_cluster.values())
	new_centroid = {i: None for i in range(len(assigned_clusters))}
	for m,n in enumerate(assigned_clusters):
		index = [i for i,j in old_cluster.items() if j == n]
		new_centroid[m] = np.mean(main_data.iloc[index, :-1].values, axis=0)
	return new_centroid

def kmeans(centroid):
	old = centroid
	new = get_new_centroid(old)	
	iteration = 1
	while not all([np.allclose(i,j) for i,j in zip(old.values(), new.values())]):
		print iteration
		iteration += 1
		temp = new
		old = temp
		new = get_new_centroid(new)	
	return new, iteration

def sse(centroid, clusters):
	sq_error = 0
	for i in centroid:
		cluster_index = [key for key,value in clusters.items() if value == i]
		cluster_rows = main_data.iloc[cluster_index, :-1].values
		sq_error += sum(euclidean(centroid[i], cluster_rows))
	return sq_error

def accuracy(centroid, clusters):
	df = main_data.iloc[:, -1]
	df = pd.DataFrame(df)
	df.columns = ['actual']
	df['predicted'] = ''
	for i in centroid:
		cluster_index = [key for key,value in clusters.items() if value == i]
		classes = df.iloc[cluster_index, 0].values
		val = Counter(classes).most_common(1)
		df.iloc[cluster_index, 1] = val[0][0]
	count = 0
	for i in df.index.values:
		if df.loc[i, 'actual'] == df.loc[i, 'predicted']:
			count += 1
	return float(count)/len(df.index)*100

if __name__ == '__main__':

	#iris: k=3
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

	method = sys.argv[1]
	main_data = pd.read_csv(url, header=None)
	indexes = main_data.index.values
	k = 3
	# random.seed(1)
	centroid_index = random.sample(indexes, k)
	centroid_data = {i: None for i in centroid_index}
	for i in centroid_data:
		centroid_data[i] = main_data.iloc[i, :-1].values
		centroid_data[i] = centroid_data[i].astype(np.float64)

	data = main_data.iloc[:, :-1].values
	iteration = 0
	final_centroids, iterations = kmeans(centroid_data)

	print "Final Centroids Points: ", final_centroids
	print "Final Clusters: ", allocate_cluster(final_centroids)
	print "Total Iterations: ", iterations
	print "Sum Squared Error: ", sse(final_centroids, allocate_cluster(final_centroids))
	print "distance calculations: ", (iterations)*len(data)*k
	print "Final Accuracy: ", accuracy(final_centroids, allocate_cluster(final_centroids))

