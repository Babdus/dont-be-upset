import os
import sys
from random import randint
import pandas as pd
from surat import Surat
from time import time
import math

def init(p, dir_path, file_from, file_to):
	dir_path_inner = dir_path + '/' + str(p)
	dir_path_inner_sources = dir_path_inner + '-sources'

	m = int(512/p)

	X = []
	Y = []

	start = time()

	file_from += int((math.log(p, 2)-4))*600
	file_to += int((math.log(p, 2)-4))*600

	for i in range(file_from, file_to):
		filename = str(i) + '.png'
		print(f'\033[33;1m{filename}\033[0m')
		image = Surat(dir_path_inner_sources + '/' + filename, m, p)

		for n1 in range(m*m):
			print(f'n1: {n1}', end='\r')

			# left edge
			left_neighbour_ns = {}
			left_neighbour_n = n1-1 if n1 % m != 0 else None
			if left_neighbour_n is not None:
				left_neighbour_ns[left_neighbour_n] = 1
			for i in range(m):
				not_left_neighnour_n = randint(0, m*m)
				if not_left_neighnour_n != left_neighbour_n:
					left_neighbour_ns[not_left_neighnour_n] = 0

			for lnn in left_neighbour_ns:
				pair = image.get_pair(lnn, n1)
				edge_vec_1 = Surat.get_edge_vector(pair, p, left=True)
				edge_vec_2 = Surat.get_edge_vector(pair, p, left=False)

				x = edge_vec_1 + edge_vec_2

				X.append(x)
				Y.append(left_neighbour_ns[lnn])

			# bottom edge
			bottom_neighbour_ns = {}
			bottom_neighbour_n = n1+m if n1 < m*(m-1) else None
			if bottom_neighbour_n is not None:
				bottom_neighbour_ns[bottom_neighbour_n] = 1
			for i in range(m):
				not_bottom_neighnour_n = randint(0, m*m)
				if not_bottom_neighnour_n != bottom_neighbour_n:
					bottom_neighbour_ns[not_bottom_neighnour_n] = 0

			for bnn in bottom_neighbour_ns:
				pair = image.get_pair(n1, bnn, left=False)
				edge_vec_1 = Surat.get_edge_vector(pair, p, left=True)
				edge_vec_2 = Surat.get_edge_vector(pair, p, left=False)

				x = edge_vec_1 + edge_vec_2

				X.append(x)
				Y.append(bottom_neighbour_ns[bnn])

	pd.DataFrame(X).to_csv('X.csv')
	pd.DataFrame(Y).to_csv('y.csv')

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