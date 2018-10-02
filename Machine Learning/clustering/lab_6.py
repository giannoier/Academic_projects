import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering

#Load data.
data = pd.read_excel('Lab6_priorities_perifereion.xlsx',index_col = 0, header=1)

#Choose between lines or columns.
#data = data.transpose()
X = data.as_matrix()

#Apply PCA to reduce dimensionality.
pca = PCA(n_components=3)
X = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
print(X)

#Use k-Means clustering.
kmeans = KMeans(n_clusters=7, random_state=0).fit(X)
print(kmeans.labels_)

#Use agglomerative clustering.
hierarchical = AgglomerativeClustering(n_clusters=7).fit(X)
print(hierarchical.labels_)
