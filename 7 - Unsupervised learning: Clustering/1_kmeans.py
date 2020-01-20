import matplotlib.pyplot as plt

# Importing the make blob module from the samples generator: make blob is an useful function to generate
# blob of points with a gaussian distribuition (blob of points concentrated around a circle, as a gaussian bell
# seen from the above)
from sklearn.datasets import make_blobs

# Let's make our actual blob: we need to pass some parameter in it:
# 1) n_samples: numbers of samples the dataset will contains (a row)
# 2) n_features: setted to default to 2: it indicated the numbers of features our dataset have.
# 3) centers: the numbers of blob we'd like to create
# 4) cluster_std: the standard deviation of a cluster
# 5) random_state: as usual, set to a number to avoid random changes in multiple iterations.
# NOTE: make_blobs will return two numpy array: the features array and the target one. Since we're in the unsupervised
# learning, we do not need any target, so let's omit the Y variable.
X, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=.8, random_state=0)

# Let's visualize our just created blob dataset. We'll use a scatterplot passing as the input features
# the first column as the x, and the second as the y. We'll pass also a S variable that indicates the size of the dots.
plt.scatter(X[:,0], X[:,1], s=20)

# Let's build our actual model: the Kmean. We briefly remember, that the KMean is an algoirthm used in unsupervised learning
# to automatically detect the class partitioning: note that, each class will be UNLABELED becase we do not know what are we
# analyzing. The idea before the KMean is really simple: all we need to do, is to choose a number of initial cluster; we then
# select K random (usually) examples to represent a centroid. Once we got our cluster, what the KMean do are the following steps:
# For each example of the dataset it will compare the current example to every centroid of the just created cluster: if the distance
# from that example to the centroid (usually measured as a norm2 - euclidean distance - or using the variance/std) is lesser than the
# actual lesser distance (initially this distance is setted to +INFINITY) it will then update the "potential cluster" as the one with
# the lesser distance. The algorithm will follow this process for all the clusters, finding who's effectively the less distant from
# the example. Done that process for each example in the dataset, it will then proceed to compute the new centroid (by analizing the
# standard deviation of the entire set) and start again analizing all the example and comparing them to the new formed cluster centroid.
# if nothing moves, then we done, otherwise the algorithm will continue until max iterations are reached or convergence has been met.
# Let's import and instantiate the model from sklearn
from sklearn.cluster import KMeans

# But what would happen if two random centroids are choosen TOO CLOSE to each other? to avoid that problem, we pass in the init argument the
# string "k-means++", a better version of the kmean that will avoid that problem: we also pass the initial number of clusters, let's say 4.
kmean = KMeans(init='k-means++', n_clusters=4)

# we can now proceed to apply the kmean and fit our data: since this isn't supervised classification, we do not need a train/test set.
kmean.fit(X)

# Let's fetch the predictions: the output will be an array containing, for each example in the dataset, the belonging class.
Y = kmean.predict(X)

# For a better visualization, let's fetch for each cluster the centroids
centroids = kmean.cluster_centers_

# Let's use both the initial features dataset and the centroids to visualize the cluster suddivision; we'll use the scatterplot function
# to achieve that, passing initially the features (as we did above in the initial plotting) and now, also passing the Y array containing the
# cluster memeberships: matplotlib will automatically assign a color based on which cluster they belong. We do the same thing to the centroids
# array, containing values in form [X,Y], acting as the two features of the set.
plt.scatter(X[:,0], X[:,1], s=20, c=Y)
plt.scatter(centroids[:,0], centroids[:,1], c='red', s=50, alpha=0.5)

# We basically done.
# But, what if we'd like to AUTOMATICALLY find the number of clusters for our datasets? Well, there is a special technique to do that.
# We use the ELBOW GRAPH to acknowledge the best possible clusters number. The elbow graph is nothing else that a graph that put, on the
# X axis the number of clusters used, and on the Y axis something called SSD: Sum of squares distances, the sum of each squared distance from
# an example to his centroid of a specific cluster. To achieve that, let's say we'd like to know which is the best number of cluster for our example,
# let's say between 0 and 10.
# Define an empty dict that will be used to store, the SSD of the current analyzed KMean model with K cluster.
SSD = {}

# for 1 to 10 cluster
for k in range(1,11):
    # create a kmean model with k clusters
    kth_kmean = KMeans(init='k-means++', n_clusters=k)

    # fit the model to the data to calculate clusters
    kth_kmean.fit(X)

    # use the .inertia_ variable to fetch the SSD for that cluster
    SSD[k] = kth_kmean.inertia_

# once done, we got our dictionary containings the sum of squaed distances for each kmean model with K cluster. To plot the graph containing
# the elbow informations, let's use the KEYS of the dictionary (the indexes representing the number of clusters) as the X axis, and the
# values for that keys (SSD) as the Y axis.
# Since our dictionary is a complex data type, we need to pass the values as a list to the plot function. We then call the list() typecast
# on both keys and values. We then build the graph and showing it.
# Note that: the BEST number of cluster is represented by the "Elbow point", where the linear curve starts to grow exponentially.
plt.plot(list(SSD.keys()), list(SSD.values()), marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("SSD")
plt.show()
