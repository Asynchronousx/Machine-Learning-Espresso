import pandas as pd

iris_nan = pd.read_csv("MLData/iris_nan.csv")
iris_nan.head()

Y = iris_nan["class"].values
X = iris_nan.drop("class", axis=1)

# Our iris dataframe presents some NaN values, and we need to fix that.
# We got some methods to apply on a pandas dataframe:

# 1: Drop records presenting a NaN value: We can achieve that with dropna, which
# will drop all the records presenting a NaN value inside.
# With dropna() executed, it will remove all the records (rows) presenting a Nan.
iris_nan.dropna()

# 2: A more intrusive method is to use dropna for each row/column that present
# a NaN value. We can drop an entire column presenting a NaN value by using dropna
# and specifying the axix: 0 for the row, 1 for the column. In this case, it will
# then drop the petal_lenght column.
iris_nan.dropna(axis=1)

# 3: A better method is to REPLACE NaN value with another one, that usually match
# Mean, Median or the Mode. Let's see all of them:

# MEAN - We calculate the mean of the iris_nan dataframe, and the use the method
# fillna passing the mean to fill the NaN value with the average. Note that,
# using mean on the entire dataframe will return a Series (dataframe) containing,
# for all the labels the mean of all their values. We then use this series with
# fillna() that will fill each NaN with the appropriate value based on the label
# they appear to be NaN in.
mean_replace = iris_nan.mean()
iris_nan.fillna(mean_replace)

# MEDIAN - The median is the "middle" value of a specific range of values.
# The median() function works exactly like mean(), it will return a series that
# will be used by fillna() to replace the missing NaN values.
median_replace = iris_nan.median()
iris_nan.fillna(median_replace)

# MODE - The mode is just the element that appears the most into a set of elements.
# For example, given the array 3,7,9,13,18,18,24 his mode would be 18 cause it's
# the element that appears the most. With each value being unique, there will be
# no mode. the function mode() will return an entire dataframe composed by, the
# first row as the mode (if present) and the others as NaN. We then need to access
# just the first row of this dataframe, and we can do that by using ILOC (that
# works by indexing) using 0 as argument to indicate the first row. We then use
# fillna to replace the values.
mode_replace = iris_nan.mode().iloc[0]
iris_nan.fillna(mode_replace)

# For the numpy array we use another simple method: The Imputer. An imputer is just
# a tool to fill missing values inside of a numpy array. We need to import it as
# follow: From sklearn 0.22 onward we need to import SimpleImputer since imputer
# has been deprecated.
from sklearn.impute import SimpleImputer
import numpy as np

# we then create an imputer object: We need to specify two things:
# 1) Strategy: could be mean, median or mode. Works exactly like the previous
#    examples.
# 2) Missing values: we need to pass the nan type, specifiied by np.nan.
imputer = SimpleImputer(strategy="mean", missing_values=np.nan)

# We then use fit_transform: as we already know, fit_transform is a combination by
# both function fit and transform. It initially calculate the mean/median/mode with
# the function FIT (X' = X - Mean / STD) and then will TRANSFORM all the np.NaN
# values into the argument passed (could be a dataframe) returning a numpy array
# with all the nan filled. 
X_imputed = imputer.fit_transform(X)
X_imputed
