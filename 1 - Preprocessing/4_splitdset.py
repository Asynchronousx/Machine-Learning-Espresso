import pandas as pd
import numpy as np

# Here we're accessing a dataset through sklearn dataset modules.
from sklearn.datasets import load_boston

# We load the boston dataset into a specific variable
boston = load_boston()

# Now, our boston variable hold the dataset: this dataset is different from the
# pandas one, since it's not a dataframe but a special sklearn data structure that
# resembles a dictionary: it is composed by a bunch of tuples.
# for having a description on that dataset we use the followind function
boston.DESCR

# Now, if we would like to transform our sklearn dataset to a pandas dataframe we
# need to call the function dataframe of pandas as follow:
boston_df = pd.DataFrame(data=np.c_[boston['data'], boston['target']], columns=np.append(boston['feature_names'], 'TARGET'))

# Now, we need this dataset expressed as a numpy array. Since the boston datasets
# is composed by 3 elements: Target, Data and Feature Names, we split them into
# 3 different numpy arrays.
# The data contain ONLY THE RAW DATA ITSELF, in form of number.
# The target contains the target values needed to train the models
# Features names contains data names.
X = boston.data
Y = boston.target

# Additionally we can load the features name into a string array with that func:
Z = boston.feature_names

# Now, we need to split the dataset. Sklearn offers a pretty easy way to do that:
# we'll import the train split module as follow
from sklearn.model_selection import train_test_split

# Check the shape of the datasets: X = (506,13) - Y = (506,0).
X.shape
Y.shape

# Now, we need to split each dataset (data, target) into two subset: Test & Train.
# We'll use python's powerful syntax language to create 4 arrays (2 for test, 2 for train)
# and usuing the train_test_split function that will return 2*N subarrays, where
# N are the datasets passed in input.
# The function takes in input the datasets we want to split and various options:
# the one we need is "test_size", that indicates the size percentual we would like
# to assign to the test set.
# It's good practice to use 70% for train and 30% for test in case of small dataset
# (< 10.000 examples).
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Check the new shape
X_train.shape
X_test.shape

# Now, if we would like to split the dataframe instead of a numpy array, things change:
# Operating with panda, we use the "sample" function that, as we remember, returns a
# shuffled samples of the values inside the dataframe. We pass 0.3 as fraction to return
# the 30% inside the test dataset (or 0.7 in case of the train one).
boston_df_test = boston_df.sample(frac=0.3)

# Now we need to build the train dataset. We can't just sample the 30% of the values
# inside the dataframe because otherwise, there will be duplicate.
# We then use the indexes of all the elements of the boston dataframe test set as
# an argument to pass inside the drop function to delete all the values already
# inside the test dataset.
boston_df_train = boston_df.drop(boston_df_test.index)

# Check the shape of the set
boston_df_test.shape
boston_df_train.shape
