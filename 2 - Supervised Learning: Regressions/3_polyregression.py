import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

boston = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                     sep='\s+',
                     names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PRATIO","B","LSTAT","MEDV"])
boston.head()

# We already worked with the boston dataset: We know that, using MEDV as the target,
# the most correlated property are RM (room numbers) and LSTAT (poverty meter) as
# correlation (direct or inverse). For the sake of simplicity and visualizing, let's
# choose from the correlation matrix other 3/4 property to show up: we'll chose them
# by their correlation value (2 direct, 2 inverse).
boston.corr()

# Creating an array containing the property which present the higher tax of correlation, both
# direct and inverse
corr_cols = ["RM", "LSTAT", "ZN", "DIS", "TAX", "INDUS", "MEDV"]

# Using seaborn, we're going to create an heatmap of those columns only
sns.heatmap(boston[corr_cols].corr(),
            xticklabels=boston[corr_cols].columns,
            yticklabels=boston[corr_cols].columns,
            annot=True,
            annot_kws={"size":12})

# Now, using seaborn once again, let's create a pairplot to put in evidence the correlation
# between each property of the subset between them.
# Observing the pairplot (focussing our attention on the last row, where MEDV (our target) is
# stored) we can observe (as we already know) that the most influent and correlated property to
# MEDV are RM and LSTAT. RM present a linear structure and could be described be a simple linear
# regression. Observing LSTAT however, it shows up a NON-LINEAR correlation with MEDV: The graph
# indeed describe a curve. And we can achieve a curve with a Polynomial regression.
sns.pairplot(boston[corr_cols])

# To build a polynomial regression, we must use the tool
from sklearn.preprocessing import PolynomialFeatures

# Now we need actually something to pass into the polyfeats transformation: as we
# did into the multiple linear regreession with the standardization, we're going to
# polyfit those value from the train and test set of the property. Let's then define our
# initial X,Y datasets and then split them up: Since we're using PolynomialFeatures to
# do a polynomial regression, we'll do that just on the LSTAT property, since it present
# the most accentuated NON-LINEAR structure. Later on, we'll do that on the entire dataset.
# IMPORTANT NOTE: When building our property array, even if we need just ONE Property we
# still need to init it as a 2D ARRRAY, because the fit functions need a 2D array as argument.
# We can achieve that by passing an array containing only the name of the property we'd like to
# analyze or, a good practice is to use drop followed by the columns we'd like to drop.
X = boston.drop("MEDV", axis=1)
Y = boston["MEDV"].values

# let's split into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=420)

# Now that we got our splitted datasets, we must define WHICH polynomial grade we would like to reach
# while computing the polynomial regression: let's put the entire training into a for cycle, and at the end
# of each cycle let's print the values of the R2 scorse and the mean error to check, which is the best suited
# degree for the best optimization.
# SPOILER: as the current boston dataset state, the most fine polynomial degree for the best result is 9, with a
# R2 score of 0.74486. The 10th iteration got 0.72597, which is lower.

for i in range (1,15):

    # passing i as degree
    polyfeats = PolynomialFeatures(degree=i)

    # and now let's fit (and add the new polynomial feature) to the train and test model
    X_train_poly = polyfeats.fit_transform(X_train)
    X_test_poly = polyfeats.transform(X_test)

    # In the polynomial arrays now will be contained the additional Nth degree specified
    # into the creation of the polynomial feature.
    # Let's now build our linear model as we already know.
    lr = LinearRegression()
    lr.fit(X_train_poly, Y_train)
    Y_pred = lr.predict(X_test_poly)

    # Testing the goodness with value checking (R2, MSE, coefficent and bias)
    print("Iteration {} -> R2 Score: {} - MSE: {}".format(i, r2_score(Y_test, Y_pred), mean_squared_error(Y_test, Y_pred)))

# Now, instead of doing the polynomial regression just on LSTAT (the NON-LINEAR property) let's
# do that on the entire dataset. Let's then build again the vectors
Xe = boston.drop("MEDV", axis=1).values
Ye = boston["MEDV"].values

# Split the train/test set
Xe_train, Xe_test, Ye_train, Ye_test = train_test_split(Xe, Ye, test_size=0.3, random_state=420)

# As before, lets put the entire model training and evalutation into a for loop to check which degree
# suits better (Spoiler: since the entire datasets got a lot of property, polynomial regression will act
# bad starting from degree=2: this because on many values polynomial regression trend to INCREASE the failure
# and decrease the accuracy of a model).
# NOTE: add standardadization of the model for a better accuracy, since there are a lot of property different in
# values (much bigger and much smaller)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

for i in range (1,5):

    # passing i as degree
    polyfeats = PolynomialFeatures(i)

    # Adds polynomial features to the current train dataset
    Xe_train_poly = polyfeats.fit_transform(Xe_train)
    Xe_test_poly = polyfeats.transform(Xe_test)

    #Standardize
    Xe_train_poly_std = ss.fit_transform(Xe_train_poly)
    Xe_test_poly_std = ss.transform(Xe_test_poly)

    #Let's now build our linear regression model to work on the polynomial train and test set
    lre = LinearRegression()
    lre.fit(Xe_train_poly_std, Y_train)
    Ye_pred = lre.predict(Xe_test_poly_std)

    # Checking the goodness of the trained model
    print("Iteration {} -> R2 Score: {} - MSE: {}".format(i, r2_score(Ye_test, Ye_pred,), mean_squared_error(Ye_test, Ye_pred)))
