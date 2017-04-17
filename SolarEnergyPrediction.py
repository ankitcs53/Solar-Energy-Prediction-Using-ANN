"""	
	Author: Ankit Choudhary, Vishal Chauhan

	Description:This code uses the pre-processed data obtained using "baseline.py" file.
		    	For each segment,
				The data is stored in CSV files as 2-d matrix having dimensions (NumberOfDays, 16 )
				First 15 columns represent the features value and last column represent the predicted output
				Using this type data we can train the neural network to predict the Solar Energy Received
				The mean_absolute_error is calculated for each segment and returned. 
				Finally we take the mean of the errors from different segments
"""

import numpy as np
np.random.seed(1337)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers
from keras.utils import np_utils


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

batch_size = 10
nb_classes = 1
nb_epoch = 70000

dataPath = "/home/ankit/Documents/Deep Learning/Dataset/"

def NeuralNetwork(X_train,Y_train,X_test,Y_test):

	# applying the principal component analysis to reduce the number of features
	pca = PCA(n_components=7)
	pca.fit(X_train)
	X_train = pca.transform(X_train)
	pca.fit(X_test)
	X_test = pca.transform(X_test)	


	#normalizing the input values to fall in -1 to 1
	X_train = X_train/180000000.0
	X_test = X_test/180000000.0

	
	model = Sequential()
	model.add(Dense(15, input_shape=(7,)))
	model.add(Activation('tanh'))

	model.add(Dense(11))
	model.add(Activation('tanh'))

	model.add(Dense(1))

	model.summary()
	sgd = optimizers.SGD(lr=0.1,momentum=0.2)
	model.compile(loss='mean_absolute_error',
	              optimizer=sgd,
	              metrics=['accuracy'])

	history = model.fit(X_train, Y_train,
	                    batch_size=batch_size, epochs=nb_epoch,
	                    verbose=1, validation_data=(X_test, Y_test))


	score = model.evaluate(X_test, Y_test, verbose=0)
	error = score[0]

	plt.plot(history.history['loss'])
	plt.show()
	plt.gcf().clear()

	return error



def main():

	errorList = []

	# for data in each of the segment
	for i in range(0,4):
		data = np.genfromtxt(dataPath+"inputnn" + str(i+1) + ".csv",delimiter=',',skip_header=0,dtype=float)
		X_train = data[:,:15]
		Y_train = data[:,15]

		data = np.genfromtxt(dataPath+"inputnn" + str(i+1) + "_test.csv",delimiter=',',skip_header=0,dtype=float)
		X_test = data[:,:15]
		Y_test = data[:,15]		

		#train and test nueral network for this segment
		error = NeuralNetwork(X_train,Y_train,X_test,Y_test)
		errorList.append(error)	
				
	
	#compute the mean error from all the segments
	print errorList
	meanerror = 0.0
	for i in errorList:
		meanerror += i
	meanerror /= 4
	print meanerror

	
if __name__ == "__main__":
    main()