import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# As we did in the kmean algorithm file, here we're importing a special kind of dataset generator named make_moons.
# the main difference between make blobs, is that this kind of model will create specific dataset NOT SOLVABLE with
# a generic clustering approach such as the KMean, because the spreading of the data through the dataset will assume
# the aspect of half moons; we'll see better later when plotting.
from sklearn.datasets import make_moons

# make an example dataset using make_moons: the arguments are similar to the make_blobs function: we pass in the number of
# examples we want to hold in, the NOISE (how much we'd like to have outliers to our data) and as usual, random state.
# Note: As kmean, since we're approaching the unsupervised learning, we do not need the labels array.
X, _ = make_moons(n_samples=200, noise=0.1, random_state=0)

# let's plot the dataset; using the same approach used in the kmean dataset visualization, since we got 2 properties into
# this dataset, we'll use the first column to represent the x axis and the second to represent the y. As we can see plotting
# this dataset, we do not have a gaussian distribuition but more likely an half circle.
plt.scatter(X[:,0], X[:,1], s=20)

# Before using DBSCAN, let's train a kmean model to check how good results are
# using the elbow method, let's select the best number of datasets, then train and displaying results.
SSD = {}
for k in range(1,11):
    kth_kmean = KMeans(init='k-means++', n_clusters=k)
    kth_kmean.fit(X)
    SSD[k] = kth_kmean.inertia_

# visualize the elbow graph
plt.plot(list(SSD.keys()), list(SSD.values()), marker='o')

# And let's then choose the best value: for this dataset, the elbow is centered in 4 cluster.
# Let's try with 2, since we got two half moon that we'd like to approximate.
kmean = KMeans(init='k-means++', n_clusters=2)
kmean.fit(X)
Y = kmean.predict(X)

# plot the scatter graph showing the suddivision made
centroids = kmean.cluster_centers_
plt.scatter(X[:,0], X[:,1], c=Y, s=20)
plt.scatter(centroids[:,0], centroids[:,1], c='red', s=40, alpha=0.5)
plt.show()

# As we can see, with the KMean algorithm we got poor results; that's mainly because the kmean can only cluster data
# in concentrics radius, showing off a limit on dataset that presents shapes different from a circle.
# For that kind of problems, we use DBSCAN (acronym of Density Based Special Clustering of Application with Noise).
# DBSCAN is one of the most powerful tool when coming to unsupervised learning through clustering. The main difference with an algorithm
# that use the KMean approach, the DBSCAN will find AUTOMATICALLY the cluster number inside a dataset. How? we do not need the
# cluster number parameter anymore, but we now need to set two parameter: Eps and minPoints.
# Eps is simply the radius that will represent the maximum distance at which the research of adjacent examples will be done.
# As usual, we consider an example and calculate the distance between that very example and the adjacent; if the distance (calculated
# with the euclidean formula for example) is lesser than the EPS value, the adjacent and the current node will be considered as potential
# memeber of a cluster.
# minPoints then, is the minimum number of points that a cluster needs to be formed. The minimum points possible are given by the formula
# numDim + 1, where numDim represents the properties and target of a given set. For the most possible simple dataset, containing ONE property
# and one target, the minPoints will be equal to 3. (numDim = 2 then 2+1 = 3). We'll call CORE POINTS all the points that forms a cluster
# (respecting then the minPoints bounds constrained by the Eps radius), BORDER POINTS all the point in the radius of a core point but NOT a
# core point themselves and NOISE POINTS all the points that are NOT included into a radius of a core point. For this extremely useful consequence,
# the noise points (or outliers) will be isolated and highlighted as possible ANOMALY in the dataset: infact, the DBSCAN algorithm is often used
# to do ANOMALY DETECTION, since it server well to this scope isolating the outliers. Also, one of the MOST important feature of the DBSCAN is that
# with this approach, it will create clusters that can have a DIFFERENT SHAPE from a simple circle, and that is useful for data that do not presents
# a gaussian distribuition (such as the one we made above), resulting in a very good approximation of clusters with different shape than a simple circle. 
# Let's implement DBSCAN through the apposite sklearn module
from sklearn.cluster import DBSCAN

# We train DBSCAN as the same way we train the KMean model: instead of the cluster number however, we need to pass the two
# parameter that will take care of the cluster formations and rules reviewing: the Eps value (the maximum distance from a point from
# a core) and the minPoints (the minimum points necessary to be in the Eps radius from the current analyzed example to consider the
# example a core point).
dbscan = DBSCAN(eps=0.20, min_samples=3)
dbscan.fit(X)
Y = dbscan.fit_predict(X)

# plot the results
plt.scatter(X[:,0], X[:,1], c=Y, s=20)
