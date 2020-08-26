from random import seed
from random import randrange
from csv import reader
from math import sqrt

def load_csv(filename):
	ds = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			ds.append(row)
	return ds

def str_column_to_float(ds, column):
	for row in ds:
		row[column] = float(row[column].strip())
 

def str_column_to_int(ds, column):
	class_values = [row[column] for row in ds]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in ds:
		row[column] = lookup[row[column]]
	return lookup
 

def ds_minmax(ds):
	minmax = list()
	for i in range(len(ds[0])):
		column_val = [row[i] for row in ds]
		value_min = min(column_val)
		value_max = max(column_val)
		minmax.append([value_min, value_max])
	return minmax

def normalize_ds(ds, minmax):
	for row in ds:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 

def cross_validation_split(ds, n_folds):
	ds_split = list()
	ds_copy = list(ds)
	fold_size = int(len(ds) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(ds_copy))
			fold.append(ds_copy.pop(index))
		ds_split.append(fold)
	return ds_split
 
# Calculate accuracy percentage
def acc_metric(actual, pred):
	crr = 0
	for i in range(len(actual)):
		if actual[i] == pred[i]:
			crr += 1
	return crr / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(ds, algorithm, n_folds, *args):
	folds = cross_validation_split(ds, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		pred = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = acc_metric(actual, pred)
		scores.append(accuracy)
	return scores

def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
 

def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors
 
# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction
 
# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
	predictions = list()
	for row in test:
		output = predict_classification(train, row, num_neighbors)
		predictions.append(output)
	return(predictions)
 
# Test the kNN on the Iris Flowers ds
seed(1)
filename = 'final_ar.csv'
ds = load_csv(filename)
for i in range(len(ds[0])-1):
	str_column_to_float(ds, i)
# convert class column to integers
str_column_to_int(ds, len(ds[0])-1)
# evaluate algorithm
n_folds = 10
num_neighbors = 7
scores = evaluate_algorithm(ds, k_nearest_neighbors, n_folds, num_neighbors)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % ( sum(scores)/float(len(scores))))