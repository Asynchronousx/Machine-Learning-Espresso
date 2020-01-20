import pandas as pd

# Pandas standard dataset are called "Dataframe", a special structure organized and mantained
# by the pandas framework. With pandas dataframes, a lot of the most complex functions are
# implemented internally to aid us with the ease of us.
# Let's see how to load, displaying and doing basic operation on the pandas dataframe structure.

# NOTE: Preliminar concepts:
# Feature/Property/Example -> the actual data of our dataset, represented by a ROW of the dataset,
# minus the target (could be more than one column) class values
# Targets -> The target features that at train time will need to be predicted. Target are predicted
# using special kind of dataset called Train Set, in which we'll store for a given property the expected
# target values. We'll see more on that in the later examples.

# loading a dataframe from the file iris.csv: Be sure that the file to be readed is into the current
# working directory, or the method will return an error.
# If wanted, the folder MLData on the github repo contains some CSV dataset ready to be used.
# Download that folder and put that into your cwd.
# to assure in which directory you are, use os.getcwd() and os.chdir("/path/to/dir") to change to the
# directory in wich MLData is located.
iris = pd.read_csv("MLData/iris.csv")

# function head/tail returns the first/last 5 element of the dataframe. Passing an
# argument to those funciton (numeric) will cause to increase the number of values displayed.
iris.head()
iris.tail()

# Method that will return infos about the dataframe: first and last 5 records, columns name,
# size etc.
iris.info

# fetch columns number
iris.columns

# Assigning one column of the dataframe to a new variable holding the dataframe and visualizing it
Y = iris["species"]
Y.head()

# Assigning more than one column to a variable that will hold the dataframe
X = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
X.head()

# Creating a copy of the dataframe
iris_sampled = iris.copy()

# Assigning a shuffling to iris_sampled. Frac = 1 means that we are taking a
# fraction equal to 1 of the dataframe, so the dataframe itself.
# I.E: setting frac=0.3 would have returned only the 30% of the dataframe.
# NOTE: the method sample will shuffle all the records inside the dataframe, returning
# a new order (but same elements). We then pass random state (a number that act as a
# randomic seed) to have constant results over multiple iteration of the dataframe sample.
# (Each number corresponds to a different seed, and different seeds will return different
# shuffling).
iris_sampled = iris.sample(frac=1, random_state=0)

# Showing off
iris_sampled.head()

# LOC & ILOC - loc use research by labels, iloc by indexes.
# accessing third element into the array using indexes showed off while using
# the head function (first no-named column) -> the number should be 33 with random_state = 0.
# if that gives error, try to display the head on your own and fetch the third index from the
# function output (usually goes from 0 to 150 on 5 columns since the dataframe shape is 150x5).
iris_sampled.loc[33]

# Accessing third element into the array by iloc using indexes
iris_sampled.iloc[2]

# Accessing a specific element caraterized by row and column
iris_sampled.loc[33, "species"]

# Accessing the second column of the first 10 elements
iris_sampled.iloc[:10, 1]

# STATISTIC DESCRIPTION

# using shape to know the matrix dimension
iris.shape

# using describe to access statistic infos
iris.describe()

# accessing a specific statistic info
iris.std()

# accessing a specifc statistic info by a property label
iris["sepal_length"].std()

# isolating property or a target, returning only the unique one
iris["species"].unique()

# Creating mask to our dataframe: pandas allow us to create masks to select only
# some of the records into a dataframe.
# This assignment will return in the mask only the petal with a lengt major than
# the mean of the petal lenght of the dataframe.
long_petal_mask = iris["petal_length"] > iris["petal_length"].mean()
long_petal_mask

# applying the mask to our dataframe to filter our results: we use the mask as as
# an index to filter out
iris_long_petals = iris[long_petal_mask]
iris_long_petals.head()

# Creating a mask that allow us to change the value of the species from "setosa"
# to undefined.

iris_copy = iris.copy()

# Here, analyze what's going on from the inside: we're accessing the array species
# through iris_copy["species"]. Then, we compare EACH element of this array with
# a string value, "setosa", and that returns a boolean: True if the analyzed
# element is == setosa, false otherwise. If true, it then continues the execution,
# assigning to the very element the value "undefined".
iris_copy[iris_copy["species"] == "setosa"] = "undefined"
iris_copy["species"].unique()

# ARITHMETIC OPERATION
# Doing arithmetic operation is easy: we need to EXCLUDE all the features into
# the dataframe that are NOT NUMERIC. That because, in ML we work on the data
# expressed as a numeric value. To do that, we drop the values that are not num
# into our dataframe: Species.

# dropping for each row record (axis=1) the species
# With axis = 1 we're saying the the entire column species must be dropped.
# axis = 0 drop an entire row.
iris.drop("species", axis=1)

# And proceding to normalize the dataframe; we're applying the normalization formula
# made by (Sum(i to n) Xi - Xmin) / (Xmax - Xmin) where X is our dataframe.
X_norm = (X - X.min()) / (X.max() - X.min())

# showing the normalization results
X_norm.head()

# Utilizing sort values function to order our dataframe by column values.
iris.sort_values("petal_length").head()

# group records by a value: using groupby specifying for which featurew we're
# gorouping by.
grouped_species = iris.groupby("species")
grouped_species.head()

# and printing mean
grouped_species.mean()

# importing numbpy module to perform operations
import numpy as np

# Using apply to use functions on row or columns:
# with AXIS=1 we'll compute the nonzero values on each value of a column
iris.apply(np.count_nonzero, axis=1).head()

# with AXIS=0 we'll compute the nonzero values on each value of a row
iris.apply(np.count_nonzero, axis=0).head()

# Using applymap to apply a function to all the values into the dataframe Using
# a lambda function

# Since we regoznie "species" at the target property, for generating a numpy array
# that contains all the features except for the targets, we must use the drop function:
# this will drop, for axis=1, the ENTIRE columns of species.
X = iris.drop("species", axis=1)

# Applymap will apply a function passed in input to the dataframe considered.
# The lambda function in our case will, for each value of the X dataset (considering
# row and column) apply a round to that very value: if the value is > 0 it will round
# the value considered at the nearest integer. If not, it will round it to 0.
# I.E: 0.55 -> 0 / 0.45 -> 0
X = X.applymap(lambda val: int(round(val,0)))

# showing off first values
X.head()

# Testing INVALID values into our dataframe and trying to correct them.
# building an invalid dataframe

iris_nan = iris.copy()

# generating a vector of 10 samples values using the randint function from numpy,
# passing as argument the first value of the shape method (that will returns the row)
# and the size that will return an array of that specified value (size=(10) will return
# a 10 elements array)
samples = np.random.randint(iris.shape[0], size=(10))

# Using the sample vector generated as a label, we're acceding to all the petal_length
# values by using samples vector for matching. There will be 10 value with None value
# after the operation.
iris_nan.loc[samples, "petal_length"] = None

# Counting invalid values: we now have our initial iris dataset but with 10 null values.
iris_nan["petal_length"].isnull().sum()

# Now, the filling comes up: we use the function fillna to fill all the blank
# values into our dataframe.

# Calculate the mean of the petal lenght not null
mean_petal_lenght = iris_nan["petal_length"].mean()

# and now fill the blank space with the mean of the petal lenght to fix the Dataframe
iris_nan["petal_length"].fillna(mean_petal_lenght, inplace=True)
# or iris_nan["petal_length"] = iris_nan["petal_length"].fillna(mean_petal_lenght)

# counting the null values: there will be 0.
iris_nan["petal_length"].isnull().sum()

# Pandas & Matplotlib
# Utilizing pandas with matplotlib to build graphics

import matplotlib.pyplot as plt

# using plot directly on the dataframe putting on the X the sepal lenght, on the
# y the sepal widht and using a scatter plot as a graphic.
iris.plot(x="sepal_length", y="sepal_width", kind="scatter")
