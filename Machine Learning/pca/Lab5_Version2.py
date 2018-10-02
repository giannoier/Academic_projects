import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def doPCA():
	from sklearn.decomposition import PCA
	pca=PCA(n_components=2)
	pca.fit(array)
	return pca



df = pd.read_excel('defra.xlsx')
#print df

array=df.as_matrix()
#print array

cov_mat = np.cov([array[:,0],array[:,1],array[:,2]],array[:,3])
print('Covariance Matrix:\n', cov_mat)


mean_x = np.mean(array[:,0])
mean_y = np.mean(array[:,1])
mean_z = np.mean(array[:,2])
mean_k = np.mean(array[:,3])

mean_vector = np.array([[mean_x],[mean_y],[mean_z],[mean_k]])

#print('Mean Vector:\n', mean_vector)


#The scatter matrix is computed by the following equation:

scatter_matrix = np.zeros((4,4))
for i in range(array.shape[1]):
    scatter_matrix += (array[i,:].reshape(4,1) - mean_vector).dot((array[i,:].reshape(4,1) - mean_vector).T)


	
# eigenvectors and eigenvalues for the from the scatter matrix
eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)

# eigenvectors and eigenvalues for the from the covariance matrix
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)


print eig_vec_cov
diagon=np.diagonal(cov_mat)



sum=np.sum(diagon)
print sum

print "From covariance matrix S, we know followings. \n\n 1. The total variance (trace of matrix) is %d " %  (sum)
for i in range(len(diagon)):
	pososto=(diagon[i]/float(sum))*100
	print " Variable X%d contributes %d/%d=%f%% " % (i+1,diagon[i],sum,pososto)


sum=np.sum(eig_val_cov)
print "\n\n 2.From Eigenvalues matrix L, we know followings."
for i in range(len(eig_val_cov)):
	pososto=(eig_val_cov[i]/float(sum))*100
	print "  The %d principal axis contains  %d/%d=%f%% " % (i+1,eig_val_cov[i],sum,pososto)	




pca=doPCA()
print pca.explained_variance_ratio_
first_pc=pca.components_[0]
second_pc=pca.components_[1]
#third_pc=pca.components_[2]
#fourth_pc=pca.components_[3]

transformed_data=pca.transform(array)
for ii,jj in zip(transformed_data,array):
	plt.scatter(first_pc[0]*ii[0],first_pc[1]**ii[0],color="r")
	plt.scatter(second_pc[0]*ii[1],second_pc[1]**ii[1],color="c")
	#plt.scatter(third_pc[0]*ii[0],third_pc[1]**ii[0],color="g")
	#plt.scatter(fourth_pc[0]*ii[0],fourth_pc[1]**ii[0],color="k")
	plt.scatter (jj[0],jj[1],color="b")

plt.xlabel("bonus")
plt.ylabel("long-term incentive")
plt.show()	

print transformed_data
