import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

daibetes = datasets.load_diabetes()
# print(daibetes.keys())
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
# print(daibetes.data)  
# # data gives in form of numbers
 
# print(daibetes.DESCR)  
# description of the data


# FOR TAKE SINGLE FEATURE AND LARGE DATASET
# diabetes_X = daibetes.data[:, np.newaxis, 2]  # taking only one feature of index 2

diabetes_X = daibetes.data  # taking all the features of the dataset


# print(diabetes_X)
# but we can't take all the features because it will give us error of shape mismatch so we take only 30 feature
diabetes_x_train = diabetes_X[:-30]  # taking all the data except last 30 data for training
diabetes_x_test = diabetes_X[-30:]  # taking last 30 data for testing

diabetes_y_train = daibetes.target[:-30]  # taking all the data except last 30 data for training
diabetes_y_test = daibetes.target[-30:]  # taking last 30 data for testing

model = linear_model.LinearRegression()  # creating model
model.fit(diabetes_x_train, diabetes_y_train)  # fitting the model with training data`

diabetes_y_predicted = model.predict(diabetes_x_test)  # predicting the data

print("Mean squared error is: ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))  # calculating the mean squared error

print("Weights: ", model.coef_)  # printing the weights
print("Intercept: ", model.intercept_)  # printing the intercept    

# plt.scatter(diabetes_x_test, diabetes_y_test)  # plotting the test data
# plt.plot(diabetes_x_test, diabetes_y_predicted)  # plotting the predicted data
# plt.show()  # displaying the plot




# then now get full data toh humne change kiya ki jo 2 index ki jagah sara data axes kiya or plot hta diya because nhi kr pate error aata
# Mean squared error is:  3035.060115291269
# Weights:  [941.43097333]
# Intercept:  153.39713623331644

# then ab new code sara data acces ke bad mean squared error kam ho gya
# Mean squared error is:  1826.4841712795046
# Weights:  [  -1.16678648 -237.18123633  518.31283524  309.04204042 -763.10835067
#   458.88378916   80.61107395  174.31796962  721.48087773   79.1952801 ]
# Intercept:  153.05824267739402

