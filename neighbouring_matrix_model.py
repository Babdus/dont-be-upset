import os
import sys
import numpy as np
import math
from time import time
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

def model(p, dir_path, file_from, file_to):
	dir_path_inner = dir_path + '/' + str(p)
	dir_path_inner_sources = dir_path_inner + '-sources'

	m = int(512/p)

	X = np.array(pd.read_csv('X.csv', index_col=0))
	Y = np.array(pd.read_csv('y.csv', index_col=0))

	print(X.shape, Y.shape)

	start = time()

	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

	print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

	classifier = Sequential()
	#First Hidden Layer
	classifier.add(Dense(24, activation='relu', kernel_initializer='random_normal', input_dim=X.shape[1]))
	#Second  Hidden Layer
	classifier.add(Dense(10, activation='relu', kernel_initializer='random_normal'))
	#Third  Hidden Layer
	classifier.add(Dense(12, activation='relu', kernel_initializer='random_normal'))
	#Output Layer
	classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

	classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

	#Fitting the data to the training dataset
	classifier.fit(X_train,y_train, batch_size=50, epochs=20)

	eval_model = classifier.evaluate(X_train, y_train)
	print(eval_model)

	y_pred = classifier.predict(X_test)
	y_pred = (y_pred>0.5)

	cm = confusion_matrix(y_test, y_pred)
	print(cm)

	file_from += int((math.log(p, 2)-4))*600
	file_to += int((math.log(p, 2)-4))*600

	for i in range(file_from, file_to):

		neighbouring_matrices = {'left': [], 'bottom': []}

		filename = dir_path + '/' + str(p) + '/' + str(i).zfill(4) + '.png'
		print(filename)

		X_left_test = []
		X_bottom_test = []

		puzzle = Surat(filename, m, p)

		for n1 in range(m*m):
			for n2 in range(m*m):

				# left
				pair = puzzle.get_pair(n1, n2)
				edge_vec_1 = Surat.get_edge_vector(pair, p, left=True)
				edge_vec_2 = Surat.get_edge_vector(pair, p, left=False)

				x = edge_vec_1 + edge_vec_2

				X_left_test.append(x)

				# bottom
				pair = puzzle.get_pair(n2, n1, left=False)
				edge_vec_1 = Surat.get_edge_vector(pair, p, left=True)
				edge_vec_2 = Surat.get_edge_vector(pair, p, left=False)

				x = edge_vec_1 + edge_vec_2

				X_bottom_test.append(x)


		X_left_test = np.array(X_left_test)
		X_bottom_test = np.array(X_bottom_test)

		print(X_left_test.shape)

		y_left_pred=classifier.predict(X_left_test)
		y_bottom_pred=classifier.predict(X_bottom_test)

		for n1 in range(m*m):

			left_row = []
			bottom_row = []
			for n2 in range(m*m):

				left_row.append(y_left_pred[n1*m*m+n2][0])
				bottom_row.append(y_bottom_pred[n1*m*m+n2][0])

			neighbouring_matrices['left'].append(left_row)
			neighbouring_matrices['bottom'].append(bottom_row)

		print(np.array(neighbouring_matrices['left']))
		print(np.array(neighbouring_matrices['bottom']))

		left_df = pd.DataFrame(neighbouring_matrices['left'])
		bottom_df = pd.DataFrame(neighbouring_matrices['bottom'])

		left_df.to_csv(filename + '_left_matrix.csv')
		bottom_df.to_csv(filename + '_bottom_matrix.csv')

	end = time()

	print(round(end-start, 3))


def main():
	p = int(sys.argv[1]) if len(sys.argv) > 1 else 64
	dir_path = sys.argv[2] if len(sys.argv) > 2 else 'data_train'
	file_from = int(sys.argv[3]) if len(sys.argv) > 3 else 10
	file_to = int(sys.argv[4]) if len(sys.argv) > 4 else 30

	model(p, dir_path, file_from, file_to)

if __name__ == '__main__':
	main()