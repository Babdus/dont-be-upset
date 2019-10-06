import os
import sys
import numpy as np
from random import randint
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
# %matplotlib inline
# import sklearn.neural_network.MLPClassifier as nn
from surat import Surat

def reverse_sequence(sequence):
	seq2 = [0] * len(sequence)
	for i, n, in enumerate(sequence):
		seq2[n] = i
	return seq2


def is_neighbour(frag1_n, frag2_n, m, reversed_sequence, left=True):
	real_place_1 = reversed_sequence[frag1_n]
	real_place_2 = reversed_sequence[frag2_n]
	if left:
		return real_place_1 + 1 == real_place_2 and real_place_2 % m != 0
	return real_place_1 + m == real_place_2


p = sys.argv[1] if len(sys.argv) > 1 else 64
dir_path = sys.argv[2] if len(sys.argv) > 2 else 'data_train(full)'

dir_path_inner = dir_path + '/' + p
dir_path_inner_sources = dir_path_inner + '-sources'

p = int(p)

m = int(512/p)

X = []
Y = []

sequences = {}

path = dir_path + '/' + 'data_train_' + str(p) + '_answers.txt'

with open(path) as file:
	name = None
	for i, line in enumerate(file):
		if i % 2 == 0:
			name = line.strip()
		else:
			sequences[name] = [int(d) for d in line.strip().split(' ')]

# print(sequences)

for file_n, filename in enumerate(os.listdir(dir_path_inner)):
	if file_n > 2:
		break
	puzzle = Surat(dir_path_inner + '/' + filename, m, p)
	image = Surat(dir_path_inner_sources + '/' + filename, m, p)
	sequence = sequences[filename]

	rev_seq = reverse_sequence(sequence)

	for n1 in range(m*m):
		print(f'n1: {n1}')

		# left edge
		left_neighbour_n = n1-1 if n1 % m != 0 else None
		# print(f'left_neighbour_n: {left_neighbour_n}')

		if left_neighbour_n is not None:
			pair = image.get_pair(left_neighbour_n, n1)
			edge_vec_1 = Surat.get_edge_vector(pair, p, left=True)
			edge_vec_2 = Surat.get_edge_vector(pair, p, left=False)

			x = edge_vec_1 + edge_vec_2

			X.append(x)
			Y.append(1)

		for i in range(m):
			not_left_neighnour_n = randint(0, m*m)
			if not_left_neighnour_n == left_neighbour_n:
				continue
			# print(f'not_left_neighnour_n: {not_left_neighnour_n}')
			pair = image.get_pair(not_left_neighnour_n, n1)
			edge_vec_1 = Surat.get_edge_vector(pair, p, left=True)
			edge_vec_2 = Surat.get_edge_vector(pair, p, left=False)

			x = edge_vec_1 + edge_vec_2

			X.append(x)
			Y.append(0)

		# right edge
		right_neighbour_n = n1+1 if (n1+1) % m != 0 else None
		# print(f'right_neighbour_n: {right_neighbour_n}')

		if right_neighbour_n is not None:
			pair = image.get_pair(n1, right_neighbour_n)
			edge_vec_1 = Surat.get_edge_vector(pair, p, left=True)
			edge_vec_2 = Surat.get_edge_vector(pair, p, left=False)

			x = edge_vec_1 + edge_vec_2

			X.append(x)
			Y.append(1)

		for i in range(m):
			not_right_neighnour_n = randint(0, m*m)
			if not_right_neighnour_n == right_neighbour_n:
				continue
			# print(f'not_right_neighnour_n: {not_right_neighnour_n}')
			pair = image.get_pair(n1, not_right_neighnour_n)
			edge_vec_1 = Surat.get_edge_vector(pair, p, left=True)
			edge_vec_2 = Surat.get_edge_vector(pair, p, left=False)

			x = edge_vec_1 + edge_vec_2

			X.append(x)
			Y.append(0)

		# bottom edge
		top_neighbour_n = n1-m if n1 >= m else None
		# print(f'top_neighbour_n: {top_neighbour_n}')

		if top_neighbour_n is not None:
			pair = image.get_pair(top_neighbour_n, n1, left=False)
			edge_vec_1 = Surat.get_edge_vector(pair, p, left=True)
			edge_vec_2 = Surat.get_edge_vector(pair, p, left=False)

			x = edge_vec_1 + edge_vec_2

			X.append(x)
			Y.append(1)

		for i in range(m):
			not_top_neighnour_n = randint(0, m*m)
			if not_top_neighnour_n == top_neighbour_n:
				continue
			# print(f'not_top_neighnour_n: {not_top_neighnour_n}')
			pair = image.get_pair(not_top_neighnour_n, n1, left=False)
			edge_vec_1 = Surat.get_edge_vector(pair, p, left=True)
			edge_vec_2 = Surat.get_edge_vector(pair, p, left=False)

			x = edge_vec_1 + edge_vec_2

			X.append(x)
			Y.append(0)

		# top edge
		bottom_neighbour_n = n1+m if n1 < m*m-m else None
		# print(f'bottom_neighbour_n: {bottom_neighbour_n}')

		if bottom_neighbour_n is not None:
			pair = image.get_pair(n1, bottom_neighbour_n, left=False)
			edge_vec_1 = Surat.get_edge_vector(pair, p, left=True)
			edge_vec_2 = Surat.get_edge_vector(pair, p, left=False)

			x = edge_vec_1 + edge_vec_2

			X.append(x)
			Y.append(1)

		for i in range(m):
			not_bottom_neighnour_n = randint(0, m*m)
			if not_bottom_neighnour_n == bottom_neighbour_n:
				continue
			# print(f'not_bottom_neighnour_n: {not_bottom_neighnour_n}')
			pair = image.get_pair(n1, not_bottom_neighnour_n, left=False)
			edge_vec_1 = Surat.get_edge_vector(pair, p, left=True)
			edge_vec_2 = Surat.get_edge_vector(pair, p, left=False)

			x = edge_vec_1 + edge_vec_2

			X.append(x)
			Y.append(0)


	print(filename, 'done')

X = np.array(X)
Y = np.array(Y)



print(X.shape, Y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(12, activation='relu', kernel_initializer='random_normal', input_dim=X.shape[1]))
#Second  Hidden Layer
classifier.add(Dense(10, activation='relu', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

#Fitting the data to the training dataset
classifier.fit(X_train,y_train, batch_size=50, epochs=10)

eval_model=classifier.evaluate(X_train, y_train)
print(eval_model)

y_pred=classifier.predict(X_test)
y_pred =(y_pred>0.5)

cm = confusion_matrix(y_test, y_pred)
print(cm)


neighbouring_matrices = {'left': [], 'top': [], 'right': [], 'bottom': []}

for file_n, filename in enumerate(os.listdir(dir_path_inner)):
	if file_n < 5:
		continue
	if file_n > 5:
		break
	print(filename)

	X_left_test = []
	X_right_test = []
	X_top_test = []
	X_bottom_test = []

	puzzle = Surat(dir_path_inner + '/' + filename, m, p)
	image = Surat(dir_path_inner_sources + '/' + filename, m, p)
	sequence = sequences[filename]

	rev_seq = reverse_sequence(sequence)

	for n1 in range(m*m):
		for n2 in range(m*m):

			# left
			pair = puzzle.get_pair(n1, n2)
			edge_vec_1 = Surat.get_edge_vector(pair, p, left=True)
			edge_vec_2 = Surat.get_edge_vector(pair, p, left=False)

			x = edge_vec_1 + edge_vec_2

			X_left_test.append(x)

			# right
			pair = puzzle.get_pair(n2, n1)
			edge_vec_1 = Surat.get_edge_vector(pair, p, left=True)
			edge_vec_2 = Surat.get_edge_vector(pair, p, left=False)

			x = edge_vec_1 + edge_vec_2

			X_right_test.append(x)

			# top
			pair = puzzle.get_pair(n1, n2, left=False)
			edge_vec_1 = Surat.get_edge_vector(pair, p, left=True)
			edge_vec_2 = Surat.get_edge_vector(pair, p, left=False)

			x = edge_vec_1 + edge_vec_2

			X_top_test.append(x)

			# bottom
			pair = puzzle.get_pair(n2, n1, left=False)
			edge_vec_1 = Surat.get_edge_vector(pair, p, left=True)
			edge_vec_2 = Surat.get_edge_vector(pair, p, left=False)

			x = edge_vec_1 + edge_vec_2

			X_bottom_test.append(x)


	X_left_test = np.array(X_left_test)
	X_right_test = np.array(X_right_test)
	X_top_test = np.array(X_top_test)
	X_bottom_test = np.array(X_bottom_test)

	print(X_left_test.shape)

	y_left_pred=classifier.predict(X_left_test)
	y_right_pred=classifier.predict(X_right_test)
	y_top_pred=classifier.predict(X_top_test)
	y_bottom_pred=classifier.predict(X_bottom_test)

	for n1 in range(m*m):

		left_row = []
		right_row = []
		top_row = []
		bottom_row = []
		for n2 in range(m*m):

			left_row.append(y_left_pred[n1*m*m+n2])
			right_row.append(y_right_pred[n1*m*m+n2])
			top_row.append(y_top_pred[n1*m*m+n2])
			bottom_row.append(y_bottom_pred[n1*m*m+n2])

		neighbouring_matrices['left'].append(left_row)
		neighbouring_matrices['right'].append(right_row)
		neighbouring_matrices['top'].append(top_row)
		neighbouring_matrices['bottom'].append(bottom_row)

	print(np.array(neighbouring_matrices['left']))
	print(np.array(neighbouring_matrices['right']))
	print(np.array(neighbouring_matrices['top']))
	print(np.array(neighbouring_matrices['bottom']))

	left_df = pd.DataFrame(neighbouring_matrices['left'])
	right_df = pd.DataFrame(neighbouring_matrices['right'])
	top_df = pd.DataFrame(neighbouring_matrices['top'])
	bottom_df = pd.DataFrame(neighbouring_matrices['bottom'])

	left_df.to_csv('left_matrix.csv')
	right_df.to_csv('right_matrix.csv')
	top_df.to_csv('top_matrix.csv')
	bottom_df.to_csv('bottom_matrix.csv')
