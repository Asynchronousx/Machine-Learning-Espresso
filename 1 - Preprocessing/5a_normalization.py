import pandas as pd

# Sometimes, we can use a remote URI to access a dataset. Since this specific dataset
# is a CSV without names, we can assign them by specifying a string array into the
# function itself. We then pass another array to specify which columns of the dataset
# we need to use.
wines = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
                    names=["Class", "Alcol", "Flavonoid"],
                    usecols=[0,1,7])

# We then create two numpy arrays: one containing the class elements, the other
# containing the entire dataframe features except from the class. We'll use the
# values method from a dataframe to convert the dataframe itself to a nump array.
X = wines.drop("Class", axis=1)
Y = wines["Class"].values

# Let's observe the the first 10 elements and check if some values are way bigger
# than other: We suspect that values into the Alcol column are  >> than flavonoid.
wines.head(10)

# We then use describe() to assure that: we see that the Min-Max range for the alcol
# is 11.03 - 14.83 and Min-Max for flavonoids are 0.34 - 5.08. A big value discrepance.
# We then need to normalize.
wines.describe()

# We can normalize using the formula: Xnorm = X - Xmin / Xmax - Xmin.
# For the sake of understanding, let's create some sub-arrays to simplify things.
# We create a copy of the wine array, that is gonna be normalized:
wines_norm = wines.copy()

# we then specify an array containing the features we want to normalize
features = ["Alcol", "Flavonoid"]

# we then create a temporary dataframe containing only those featues
to_norm = wines_norm[features]

# Now we can proceed to the normalization using the formula we stated above:
wines_norm[features] = (to_norm - to_norm.min()) / (to_norm.max() - to_norm.min())

# check for results
wines_norm.head()

# We could have done this in just one line: but it's more confusionary.
# wines[features] = (wines[features] - wines[features].min()) / (wines[features].max() - wines[features].min())
# wines.head()

# To apply normalization to a numpy array, we got an easier solution: we just use
# an already-existent method from the sklearn framework, named MinMaxScaler: as you
# could imagine, this tool use min-max values to scale the dataset to a fixed range
# values (def: 0-1), that is the normalization itself.

# Import the module and assigning it
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()

# Copy the X numpy array and apply the normalization on it: as usual, we use the
# fit transform method to gain infos with fit about min/max and then transforming
# the value into the array with transform applying the normalization.
# NOTE: for faster and cleaner code, let's use fit_transform that will do both
# of those process in the same function. Once the MinMaxScaler has been fitted and
# trained, we can just use transform on next dataset to normalize.
X_norm = X.copy()
X_norm = mms.fit_transform(X_norm)
X_norm[:5]
