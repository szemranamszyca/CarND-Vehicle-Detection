import glob
import time
import numpy as np
from feature_extract import extract_features
from sklearn.externals import joblib


color_space='YCrCb'
spatial_size=(32, 32)
hist_bins=32
orient=9
pix_per_cell=8
cell_per_block=2
file_classifier_name = 'svm_class.pkl'
file_scaler_name = 'svm_scaler.pkl'


def train_classifier():
	t=time.time()
	cars = glob.glob('vehicles/**/*.png')
	car_features = extract_features(cars, color_space=color_space,
	                        spatial_size=spatial_size, hist_bins=hist_bins,
	                        orient=orient, pix_per_cell=pix_per_cell,
	                        cell_per_block=cell_per_block, hog_channel='ALL')

	notcars = glob.glob('non-vehicles/**/*.png')
	notcar_features = extract_features(notcars, color_space=color_space,
	                        spatial_size=spatial_size, hist_bins=hist_bins,
	                        orient=orient, pix_per_cell=pix_per_cell,
	                        cell_per_block=cell_per_block, hog_channel='ALL')
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to extract features')

	# Create an array stack of feature vectors
	X = np.vstack((car_features, notcar_features)).astype(np.float64)

	# Define the labels vector
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


	from sklearn.preprocessing import StandardScaler
	# Fit a per-column scaler
	X_scaler = StandardScaler().fit(X)
	# Apply the scaler to X
	scaled_X = X_scaler.transform(X)


	# Split up data into randomized training and test set
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, stratify =y)
	print('Feature vector length:', len(X_train[0]))

	from sklearn.svm import LinearSVC
	# Use a linear SVC
	svc = LinearSVC()
	# Check the training time for the SVC
	t=time.time()
	svc.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train SVC...')


	# Check the score of the SVC
	print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
	# Check the prediction time for a single sample
	t=time.time()
	n_predict = 10
	print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
	print('For these',n_predict, 'labels: ', y_test[0:n_predict])
	t2 = time.time()
	print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

	joblib.dump(svc, file_classifier_name)
	joblib.dump(X_scaler, file_scaler_name)



