#!/usr/bin/env/ python

import numpy as np
import pandas as pd

'''
Command to execute: python filemane.py
uncomment the url corresponding to a data set to run for that particular 
data set
To switch between F(k-1) x F(1) & F(k-1) x F(k-1) and confidence and lift 
change the variable method and metric, respectively.
method1 = F(k-1) x F(1) and method2 = F(k-1) x F(k-1)
metric = confidence, for confidence-based pruning
metric = lift, for lift-based pruning
Change support and confidence/lift threshold in the line 220
'''

def get_dataset(url):
	'''
	get the required dataset
	'''
	data = pd.read_csv(url, header=None)
	for col in data.columns:
		data = data[data[col] != '?']
	data = pd.get_dummies(data)
	total_cols = data.shape[1]
	data.columns = range(total_cols)
	main_arr = []
	for i in range(len(data)):
		arr = []
		for j in range(total_cols):
			if data.iloc[i, j] == 1:
				arr.append(j)
		main_arr.append(arr)
	return main_arr

def get_c1(data):
	'''
	get the list of candidate items of size 1
	'''
	unique_items = set([j for i in data for j in i])
	unique_items = [frozenset([i]) for i in unique_items]
	return unique_items

def check_support(data, candidate, min_support):
	'''
	pruning item set with lesser support than threshold
	'''
	count = {i: 0 for i in candidate}
	for i in data:
		for j in candidate:
			if j <= set(i):
				count[j] += 1
	data_len = float(len(data))
	freq_candidate = []
	total_support = {}
	for i in count:
		supp = float(count[i]) / data_len
		if supp >= min_support:
			freq_candidate.append(i)
		total_support[i] = supp
	return freq_candidate, total_support

def get_candidate(prev_freq, k, method):
	final = []
	'''
	generate candidate k-items
	'''
	'method1: F(k-1) x F(1)'
	if method == 'method1':
		freq1_new = [j for i in freq1 for j in i]
		freq1_new = sorted(freq1_new)
		for m in prev_freq:
			for n in freq1_new:
				if n not in m and all(i < n for i in m):
					final.append(m | frozenset([n]))

	'method2: F(k-1) x F(k-1)'
	if method == 'method2':
		for m in range(len(prev_freq)):
			for n in range(m+1, len(prev_freq)):
				f1 = list(prev_freq[m])[: -1]
				f1 = sorted(f1)
				f2 = list(prev_freq[n])[: -1]
				f2 = sorted(f2)
				if f1 == f2 and list(prev_freq[m])[-1] != list(prev_freq[n])[-1]:
					final.append(prev_freq[m] | prev_freq[n])
	return final

def apriori(data, min_support):
	'''
	Apriori Algorithm for frequent item set generation
	'''
	candidates_generated = len(cand1)
	freq_candidates = len(freq1)
	freq_total = [freq1]
	k = 2
	while freq_total[k - 2] and k <= 4:
		# print freq_total[k - 2]
		cand_k = get_candidate(freq_total[k - 2], k, method)
		candidates_generated += len(cand_k)
		freq_k, supp_k = check_support(data, cand_k, min_support)
		freq_candidates += len(freq_k)
		support.update(supp_k)
		freq_total.append(freq_k)
		k += 1
	print 'candidates generated: ', candidates_generated
	print 'frequent candidates: ', freq_candidates
	return freq_total, support

def get_rules(freq_items, support, min_val):
	'''
	generate association rules
	'''
	rules = []
	for i in range(1, len(freq_items)):
		for freq in freq_items[i]:
			aset = [frozenset([i]) for i in freq]
			if i > 1:
				get_candidate_rules(freq, aset, support, rules, min_val)
			else:
				if metric == 'confidence':
					get_confidence(freq, aset, support, rules, min_val)
				if metric == 'lift':
					get_lift(freq, aset, support, rules, min_val)
	return rules

def get_confidence(freq, aset, support, rules, min_conf):
	'''
	calculate confidence of rule
	'''
	high_aset = []
	for i in aset:
		conf = support[freq] / support[freq - i]
		if conf >= min_conf:
			print list(freq - i), '----->', list(i), 'confidence: ', conf
			rules.append((freq - i, i, conf))
			high_aset.append(i)
	return high_aset

def get_lift(freq, aset, support, rules, min_lift):
	'''
	calculate lift of rule
	'''
	high_aset = []
	for i in aset:
		try:
			lift = support[freq] / (support[freq - i] * support[i])
		except:
			continue
		if lift >= min_lift:
			print list(freq - i), '----->', list(i), 'lift: ', lift
			rules.append((freq - i, i, lift))
			high_aset.append(i)
	return high_aset

def get_candidate_rules(freq, aset, support, rules, min_val):
    '''
    Generate a set of candidate rules
    '''
    m = len(aset[0])
    if len(freq) > m + 1:
        Hmp1 = get_candidate(aset, m + 1, method)
        if metric == 'confidence':
        	Hmp1 = get_confidence(freq, Hmp1,  support, rules, min_val)
        if metric == 'lift':
        	Hmp1 = get_lift(freq, Hmp1,  support, rules, min_val)
        # Hmp1 = get_confidence(freq, Hmp1,  support, rules, min_val)
        if len(Hmp1) > 1:
            get_candidate_rules(freq, Hmp1, support, rules, min_val)

def get_freq_maximal(freq_items):
	'''
	generate maximal frequent item sets
	'''
	freq_items = freq_items[:-1]
	non_maximal_freq = []
	for i in range(len(freq_items)-1):
		curr = freq_items[i]
		next = freq_items[i+1]
		for m in curr:
			for n in next:
				if m.issubset(n):
					non_maximal_freq.append(m)
					break
	freq_items = freq_items[:-1]
	freq_items = [j for i in freq_items for j in i]
	return len([i for i in freq_items if i not in non_maximal_freq])

def get_freq_closed(freq_items):
	'''
	generate closed frequent item set
	'''
	freq_items = freq_items[:-1]
	non_closed_freq = []
	for i in range(len(freq_items)-1):
		curr = freq_items[i]
		next = freq_items[i+1]
		for m in curr:
			for n in next:
				if m.issubset(n) and support[m] == support[n]:
					non_closed_freq.append(m)
					break
	freq_items = freq_items[:-1]
	freq_items = [j for i in freq_items for j in i]
	return len([i for i in freq_items if i not in non_closed_freq])

if __name__ == '__main__':

	#uncomment this to run for car data set
	# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
	
	#uncommet this to run for nursery data set
	# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data'
	
	#uncomment this to run for mushroom data set
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
	data = get_dataset(url)
	min_support, min_val = 0.05, 0.7    #change support and threshold value here
	method,  metric = 'method1', 'confidence'
	cand1 = get_c1(data)
	freq1, support = check_support(data, cand1, min_support)
	freq_items, support = apriori(data, min_support)
	print len((get_rules(freq_items, support, min_val)))
	# print freq_items
	# print support
	# print 'maximal frequent: ', get_freq_maximal(freq_items)
	# print 'closed frequent: ', get_freq_closed(freq_items)


