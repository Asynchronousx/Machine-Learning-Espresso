import pandas as pd
import numpy as np

# Loading boston dataset through site and assigning a special separator
boston = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                     sep="\s+",
                     usecols=[5,13],
                     names=["RM", "MEDV"])

boston.head()

# Now let's load in two different numpy arrays data and target: RM is the data,
# and MEDV is the target.
# since we have just two columns, we could use X = boston["RM"] to fetch our data,
# but let's operate in a standard way dropping the target column.
# NOTE: Initially, our Boston dataframe was composed by just two columns: RM and MEDV.
# Since we drop MEDV (the target) to put into X the properties (only RM in this case)
# we're assigning to X a MATRIX, composed by one element: RM. That becaue the fit functions
# (fit, transform and fit_transform) will need a 2D array into it as argument.
X = boston.drop("MEDV", axis=1)
Y = boston["MEDV"]

# Checking the shape
X.shape
Y.shape

# Splitting the dataset into train and test: we'll use the sklearn modules to do
# that, remembering that we're going to use 70% and 30% of the dataset for,
# respectively, train and test

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Checking the shape
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape

# Now the real fun begin: we need to build our actual model through simple linear
# regression. To achieve that, we use an appropriate sklearn module, called
# LinearRegression.
from sklearn.linear_model import LinearRegression

# And assign the linear regression to a dummy variable
lr = LinearRegression()

# Let's build our actual model: to achieve that, we need to actually train our
# model, and that's is easily achievable by the fit function: As we have stated
# during our tutorials, EVERY ESTIMATOR does have a fit function, in which, his
# aim, is to ONLY finding the internal parameters of a model that will be used
# to transform data. Then, when using transform, we're APPLYING those parameter
# to the data. We may also fit a model and then using transform on another one.
# Sticking to our model, a linear regression, fit will return the parameters
# known as BIAS and WEIGHT of the line equation in form: Y = B + WX.
# let's use the fit function to calculate the parameters using the X,Y train
# datasets: We're X_train because we need to pass the property of a datasets,
# and we're passing Y_train because they represent the target we want to achieve.

lr.fit(X_train, Y_train)

# Once the fit process has been finished, we're basically done: we got our
# internal hyperparamters, and can start now to predict, given a set of property
# (i.e: X_test) their target: Y_pred. Let's write that
Y_pred = lr.predict(X_test)

# Once predicted, we can now check the goodness of our prediction model by
# applying a cost function: we know that for linear regression, the most convenient
# cost function is the RSS (residual square sum). Sadly sklearn do not have RSS,
# but a variant: MSE, (mean squared error): The technique is basically the same as
# RSS: Sum(i to n) (Yi - (B - Wxi))^2 / Xmean but DIVIDED By the mean.
# let's import the moduel
from sklearn.metrics import mean_squared_error

Y_test.shape
Y_pred.shape

# and now let's calculate the MSE by passing in input the already known Y_test
# and the predicted Y_pred.
mean_squared_error(Y_test, Y_pred)

# We now got the mean squared error value: It could be big, it could be small.
# How do we determine that? We need an evalutation function. To achieve that,
# we use the Coefficent of Determination, a value in range 0-1 to check how good
# our prediction was. The more near to 1 -> prediction good / The more near to 0
# -> prediction bad.
# It could also return negative values for prediction made on the test set.
# Careful: this is not a cost function, but a score function: the more near to 1,
# the more good the prediction is.

from sklearn.metrics import r2_score

# And now test the accuracy of the model based on what we used to calculate the
# MSE:
r2_score(Y_test, Y_pred)

# For further information, let's check the angular coefficent (w) and the bias (b)
# built from the function fit when we used fit to create the model.
# Note that, .coef_ is an array composed by N element (N number of parameters)
# and we then need to access the first one because we got just one parameter.
lr.coef_[0]
lr.intercept_

# Let's now starting building graphs to show what we achieved:
# we'll use two scatter graph with the aux of the matplotlib
import matplotlib.pyplot as plt

# Let's build our plot with the scatter function: first using train set, both for
# X and Y, then for the test set.
plt.scatter(X_train, Y_train, c="green", edgecolor="white", label="Train Set")
plt.scatter(X_test, Y_test, c="blue", edgecolors="white", label="Test Set")

# Plot some labels for the relative axis
plt.xlabel("Average Room Numbers [RM]")
plt.ylabel("Value in 1000$ [MEDV]")

# Put the legend in the upper left location
plt.legend(loc="upper left")

# Plot the linear regression line using the X test and Y predicted datasets
# Note: the blue and green dot at the edged are the outliers.
plt.plot(X_test, Y_pred, color='red', linewidth=3)
