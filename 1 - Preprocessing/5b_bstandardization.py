import pandas as pd
import numpy as np

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
Y = wines["Class"].values
X = wines.drop("Class", axis=1)

# We're going to Standardize our dataframe. We remember that, the Standardization
# aims to create a distribuition (called Normal Distribution) of values centered in 0
# with a standard deviation value of 1.

# For the pandas dataframe we do not have a method to achieve that, so we must use
# the formula: Xstd = (X - Xmean) / Xstd

# Since we need to standardize only Alcol and Flavonoid, we create an array of
# names to pass to the wines dataframe
features = ["Alcol", "Flavonoid"]

# For the sake of simplicity we then create a copy of the dataset to work on a
# separate dataframe, and assigning to a dataframe only the columns which we're
# interested in (with the features array created before)
wines_std = wines.copy()
to_std = wines_std[features]

# We then procede to standardize with the formula
wines_std[features] = (to_std - to_std.mean()) / to_std.std()

# check the result
wines_std.head()

# For standardize the numpy array, we operate in the exact way of the normalization:
# we import a dedicated sklearn module to achieve that and assigning it to a variable
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

# creating a numpy array copy to operate with and standardize it through the
# ss variable calling fit transform, passing the just defined X dataframe standardized
# as argument (it will return a numpy array)
# NB: fit_transform will fit (train) the model to normalize the dataset and then
# will apply transform to it. 
X_std = X.copy()
X_std = ss.fit_transform(X_std)

# visualizing it
X_std[:5]
