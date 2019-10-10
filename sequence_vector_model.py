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

def model():

	start = time()

	X = np.array(pd.read_csv('X2.csv', index_col=0))
	Y = np.array(pd.read_csv('Y2.csv', index_col=0))

	print(X.shape, Y.shape)


	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

	print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

	classifier = Sequential()
	#First Hidden Layer
	classifier.add(Dense(144, activation='relu', kernel_initializer='random_normal', input_dim=X.shape[1]))
	#Second  Hidden Layer
	classifier.add(Dense(96, activation='relu', kernel_initializer='random_normal'))
	#Output Layer
	classifier.add(Dense(Y.shape[1], activation='softmax', kernel_initializer='random_normal'))

	classifier.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['categorical_accuracy'])

	#Fitting the data to the training dataset
	classifier.fit(X_train,y_train, batch_size=50, epochs=20)

	eval_model = classifier.evaluate(X_train, y_train)
	print(eval_model)

	y_pred = classifier.predict(X_test)
	y_pred = (y_pred>0.5)

	end = time()

	print(round(end-start, 3))

	# cm = confusion_matrix(y_test, y_pred)
	# print(cm)

def main():
	model()

if __name__ == '__main__':
	main()