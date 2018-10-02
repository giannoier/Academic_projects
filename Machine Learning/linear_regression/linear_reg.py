# simple excercise that implements linear regression on a restaurants data set

import numpy as np
from numpy import linalg as la
import pandas as pd

# load data
data = pd.read_csv("http://www.stat.tamu.edu/~sheather/book/docs/datasets/MichelinNY.csv")
# split into features matrix and ground truth vector
X = data[['InMichelin', 'Food', 'Decor', 'Service']].as_matrix()
Y = data[['Price']].as_matrix()
# append a column of ones in front of the features for the bias
X_ext = np.ones((len(Y),1))
X_ext = np.hstack((X_ext,X))
# calculate the transpose matrix
X_ext_trans = np.transpose(X_ext)
# solve the problem to find the parameters vector
params = la.solve(np.dot(X_ext_trans,X_ext),np.dot(X_ext_trans,Y))

# calculate the sigma squared matrix
sigma_squared = sum(((Y - np.dot(X_ext,params))**2)/(len(X_ext[:,1])-len(X_ext[1,:])))
# calculate the variance-covariance matrix
var_covar = sigma_squared * la.inv(np.dot(X_ext_trans,X_ext))
# calculate the standard error
std_error = np.sqrt(np.diag(var_covar)) 

output = pd.DataFrame(index=['constant','InMichelin','Food','Decor','Service'])
output['parameters'] = params
output['standard error'] = std_error
print output
