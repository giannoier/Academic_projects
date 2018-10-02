import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ex3 import sum_squared_error, total_sum_squares, SSR, r_squared, mean_squared_error, standard_deviation

# X: the number of disk I/Os
# Y: the processor times
# n: 7 programms

X = np.array([14,16,27,42,39,50,83])
Y = np.array([2,5,7,9,10,13,20])
n = X.shape[0]

m_X = np.mean(X)
ssX = sum(X**2)

# Fit the training data with a 1st order polynomial.
z = np.polyfit(X,Y,1)
# Add a column of ones at the end for the intercept.
Xext = np.vstack((X,np.ones(X.shape[0],)))

# Estimate using the polynomial coefs computed previously.
prediction = np.sum(Xext.T*z,axis=1)

# Compute the error and squared error for the data.
error = Y - prediction
squared_error = error**2

# Put all stuff together in a dataframe.
results = pd.DataFrame(np.column_stack((X,Y,prediction,error,squared_error)))
results.columns = ['Disk I/O', 'CPU time', 'Estimate', 'Error', 'Squared error']
print results

# Plot the data and the learned model.
p = np.poly1d(z)
plt.plot(X,Y,'ro',X,p(X))
plt.xlabel("Disk I/Os")
plt.ylabel("CPU time")
#plt.show()

sse = sum_squared_error(squared_error)
sst = total_sum_squares(prediction, Y)
ssr = SSR(sst,sse)
r_s = r_squared(ssr,sst)
print "\nSum of sqared errors:",sse
print "Total sum of squares:",sst
print "SSR:",ssr
print "R squared:",r_s

mse = mean_squared_error(sse,n)
se = standard_deviation(mse)

print "\nMean squared error:",mse
print "Standard deviation:",se

# Compute b0 and b1 standard devietions.
sb0 = se*np.sqrt((1/n) + (m_X**2)/(ssX - n*(m_X**2)))
sb1 = se/np.sqrt(ssX - n*(m_X**2))
print "\nSb0:",sb0
print "Sb1:",sb1

# Compute the 90% confidence intervals for b0 and b1.
# the 0.95-quantile of a t-variate with 5 degrees of freedom is 2.015
t = 2.015
b0, b1 = z[1],z[0]
(ci_0_low, ci_0_high) = b0 - t * (sb0), b0 + t * (sb0)
(ci_1_low, ci_1_high) = b1 - t * (sb1), b1 + t * (sb1)
print "90% Confidence Interval for b0 is:",(ci_0_low, ci_0_high)
print "90% Confidence Interval for b1 is:",(ci_1_low, ci_1_high)

# Predict the CPU time for 100 disk I/Os
num_io = 100 
x_new = [num_io, 1]
prediction = sum(x_new*z)
print "Predicted CPU time for",num_io,"I/Os:",prediction

# Standard deviation for future predictions. 
sy_p = se * np.sqrt((1/n) + ((num_io - m_X)**2)/(ssX - n*(m_X**2)))
print "The standard deviation of the predicted mean of a large number of observations is:",sy_p
(ci_mp_low, ci_mp_high) = prediction - t * (sy_p), prediction + t * (sy_p)
print "90% Confidence Interval for the predicted mean is:",(ci_mp_low, ci_mp_high)

# For a single future prediction.
s_sp = se * np.sqrt(1 + (1/n) + ((num_io - m_X)**2)/(ssX - n*(m_X**2)))
print "The standard deviation of the predicted mean for a single prediction is:",s_sp
(ci_p_low, ci_p_high) = prediction - t * (s_sp), prediction + t * (s_sp)
print "90% Confidence Interval for a single prediction is:",(ci_p_low, ci_p_high)

# Some visual tests for the model.
plt.subplot(1,2,1)
plt.plot(X,Y,'ro',X,p(X))
plt.xlabel("Disk I/Os")
plt.ylabel("CPU time")
plt.subplot(1,2,2)
plt.plot(p(X),error,'ro')

plt.show()


