import os
import sys
from random import randint
import pandas as pd
import numpy as np
from surat import Surat
from time import time
import math

def init(p, dir_path, file_from, file_to):
	m = int(512/p)

	file_from += int((math.log(p, 2)-4))*600
	file_to += int((math.log(p, 2)-4))*600

	X = []
	Y = []

	sequences = {}

	path = dir_path + '/' + 'data_train_' + str(p) + '_answers.txt'

	start = time()

	with open(path) as file:
		name = None
		for i, line in enumerate(file):
			if i % 2 == 0:
				name = line.strip()
			else:
				sequences[name] = [int(d) for d in line.strip().split(' ')]

	for i in range(file_from, file_to):
		left_matrix_path = dir_path + '/' + str(p) + '/' + str(i).zfill(4) + '.png_left_matrix.csv'
		bottom_matrix_path = dir_path + '/' + str(p) + '/' + str(i).zfill(4) + '.png_bottom_matrix.csv'

		left_matrix = np.array(pd.read_csv(left_matrix_path, index_col=0))
		bottom_matrix = np.array(pd.read_csv(bottom_matrix_path, index_col=0))

		left_vector = left_matrix.reshape(left_matrix.shape[0]*left_matrix.shape[1],)
		bottom_vector = bottom_matrix.reshape(bottom_matrix.shape[0]*bottom_matrix.shape[1],)

		vector = np.concatenate((left_vector, bottom_vector), axis=0)
		# print(list(vector))
		X.append(list(vector))

		y = sequences[str(i).zfill(4) + '.png']
		# print(y)
		Y.append(y)

	print(np.array(X).shape, np.array(Y).shape)

	pd.DataFrame(X).to_csv('X2.csv')
	pd.DataFrame(Y).to_csv('Y2.csv')

	end = time()

	print(round(end-start, 3))

def main():
	p = int(sys.argv[1]) if len(sys.argv) > 1 else 64
	dir_path = sys.argv[2] if len(sys.argv) > 2 else 'data_train'
	file_from = int(sys.argv[3]) if len(sys.argv) > 3 else 10
	file_to = int(sys.argv[4]) if len(sys.argv) > 4 else 30

	init(p, dir_path, file_from, file_to)

if __name__ == '__main__':
	main()