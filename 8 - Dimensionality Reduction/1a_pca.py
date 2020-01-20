import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

# Since column labels are a lot of them, let's use a string array to hold them all
columns = ["label", "alcohol", "malic acid", "ash", "ash alkalinity", "magnesium",
           "total phenols", "flavonoids", "non-flavonoid phenols", "proanthocyanidin",
           "color intensity", "hue", "OD280 / OD315 of diluted wines", "proline"]

# we then pass to the names parameter the just created array, so when loading CSV with read_csv,
# names will be automatically imported from that.
wines = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
                    names=columns)

wines.head()

# Let's then create the corrispective numpy arrays containing properties and target
X = wines.drop("label", axis=1).values
Y = wines["label"].values

# Note that, we got 3 DIFFERENT CLASSES of wines inside of this dataset. When reducing dimensionality, even from all those
# 13 properties to just two, this information will be mantained.
wines["label"].unique()

# Ok, we just said we got a lot of properties there, so we start to think: since those columns seems a lot, is there a way
# to reduce the "dimensionality"? (intended as the number of properties/features present inside a dataset; 13 columns
# or 13 properties equals to 13 dimension)
# The answer is YES, and that's the PCA (or Principal Component Analysis), a NON-SUPERVISED dimensionality reduction technique.
# Why not supervised? Because we reduce ONLY the properties, not caring at all for the target.
# What is, in brief, the PCA? The PCA is a particoular kind of feature extraction: we remember that a feature extraction is
# a method to reduce dimensionality in which we can express properties as linear combination of some others; in this way,
# we can reduce dimensionality from N dimension to a pleasing quantity K. We'll see how in minutes.
# How PCA works? The way in which the PCA algorithm works rely on concepts of linear algebra and statistic. Let's make it
# simple as possible. PCA makes just ONE assumption: that inside a dataset, the most important features got the higher variance.
# (And we remember that variance is a metric to indicate the values variability in a distribuition). With variance, we can measure
# and quantify the data distribution amplitude, especially on a graph where we can see clearly what's going on.
# In the dataset, will often exists properties with a LOW variance value, and that means they aren't giving much information to the
# dataset itself. Ideally, with properties presenting low variance value, we could just drop the column in which those values are stored.
# In real cases, often properties are related between them by a STRONG variance value. That means we can't just drop them. What do in those
# cases? The PCA will find (with a lot of math calculus) the direction of MAXIMUM variance in form of a line: This special line will be called
# "First Principal Component" or FPC. Found the first FPC, is possible to find N-1 more FPC, where N is the dimensionality of the dataset.
# (So, with a 13 dimension dataset, we can have 13 FPC). We got just ONE constraint applied to those line: all the First Principal Component
# of other dimension (then different from the MAXIMUM VARIANCE LINE) will be ORTHOGONAL (or normal if we consider them as vector) to the FPC
# with the highest variance. This limitation will allow us to find INDIPENDET AXIS, that means they'll have DIFFERENT VALUES between them.
# With a little bit of fantasy we can draw an example here, reducing from 2D to 1D
#
# |       \/ o                          o     /|\
# |     o /\  o                           o    |
# |   o  / o\ Second FPC  ====>      ---------------------->   ====>    ---oo---oo----------oo----->
# |     / o                                o   |       o
# |  MAX Variance FPC                  o       |     o
# |_________________
#
# As we can see, the First Principal Component with the higher vaiance is BIGGER than the ORTHOGONAL second FPC. What's that means, is that
# the second FPC will have a LESSER variance than the first. So we can discard the second component to mantain only the first. This is how
# (really approximaed) PCA works. For further information about the math behind, check some online sources.
# Let's now do that on our dataset: firs thing first, import the correct module from sklearn
from sklearn.decomposition import PCA

# BEFORE even loading the PCA module, when doing dimensionality shrinking we NEED to take data on the same scale. We can use standardization
# or normalization in base of our needs. let's use standardization.
ss = StandardScaler()
X = ss.fit_transform(X)

# Let's then create the actual PCA method: we pass as argument the number of dimensionality we want to achieve; if none are passed, then the
# algorithm will not shrink anything. Remember that this will act as a preprocessing method: will return a new array containing the results
# NOTE: we can also create a PCA module in this way:
# pca = PCA(0.95) -> the number inside the parenthesis indicated the variance value we want to keep: lower value will result in lower dimensions output,
# higher variance value in higher dimensions. We'll see that initialization in the next file, when we'll use PCA to reduce training time.
# Note that, reducing variance will DECREASE the dimensionality, but also the prediction's accuracy, so be careful and find some good comprimises.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Let's now plot what we achieved with the dimensionality reduction: now, X_pca should contain just two properties. As results, we can cleary see
# that the component, even if TWO now, still can recognize the three different target labels. We then reduced the dataset by 11 dimensionality!
# and that will help us a lot during the train phase.
plt.scatter(X_pca[:, 0], X_pca[:,1], c=Y, edgecolors="black")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()

# BETTER COMPONENT SELECTION
# Let's now try to understand how we can do a good selection on the dimensionality number.
# What if we do not know HOW MANY dimensionality are needed? In this case, we need to UNDERSTAND how they can be optimally shrinked.
# To understand the optimal number of dimensionality without losing too much information while shrinking the feautres, we need to plot
# on a graph the variance and the cumulative variance to understand better which properties give the maximum amount of information and cut the rest.
# For the clarity sake, let's build another model to achieve that, passing None as component number.
pca_graph = PCA(n_components=None)

# And let's ONLY fit the model to calculate his internal parameters, such as variance values.
pca_graph.fit(X)

# Now, with explained_variance_ratio_ we can check for each features his variance value
pca_graph.explained_variance_ratio_

# To have a better undestanding, let's graph them: we'll use both the variance and the cumulative variance for building an optimal graph that
# allow us to understand THE BEST dimensionality number for that dataset. Before doing that, let's understand what cumulative means:
# Brief example np.cumsum([1,2,3,4,5]) = [1,3,6,10,15]. A cumulative sum is a sum that, for each element will sum his predecessors (1st = 1+0 = 1,
# 2nd = 2+1 = 3, 3rd = 3+2+1 = 6, 4th = 4+3+2+1 = 10 and 5th = 5+4+3+2+1 = 15 in this specific case).
# Let's plot then the graph: To undesrstand in the best possible way, let's draw bars; We use the bar function of matplotlib passing as argument the range
# of the column we want to create range(1, MaxDimNum) AND passing the HEIGHT of those bar: those heights are contained in explained_variance_ratio_ (variance
# value for each FPC). We take adjavantage of a step plot to check how each variance contribuite to the final information provision: We'll use cumulative
# variance to explain that value. We use the where='mid' value to place each step in the middle of the bar, to understand better which is who.
# SPOILER: We can see that the 80% of information are contained into the first five FPC for the graph. So 5 Dimensionality should be the best components number
# achievable.
plt.bar(range(1,14), pca_graph.explained_variance_ratio_)
plt.step(range(1,14), np.cumsum(pca_graph.explained_variance_ratio_), where='mid')
plt.xlabel("First Principal Components")
plt.ylabel("Variance")
plt.show()
