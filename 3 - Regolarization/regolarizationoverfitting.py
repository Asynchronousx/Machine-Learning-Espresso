import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

boston = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                     sep='\s+',
                     names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PRATIO","B","LSTAT","MEDV"])

boston.head()

# This code will aim to show the problem of the overfitting over a complex dataset
# (like boston housing dataset, containing a lot of properties) and resolution to
# that very problem. To start, we're creating two numpy arrays containing the
# properties (all except from MEDV) and the target (MEDV)
X = boston.drop("MEDV", axis=1).values
Y = boston["MEDV"].values

# Going to split this dataset in two dataset: test and train for model training purpose
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=420)

# shape of the test and train dataset: (354,13) for the train set, (152,13) for the test set
X_train.shape
X_test.shape

# Recognizing the overfit is pretty easy: we need to compare the error on the train
# set and the error on the test set: if those errors presents an huge gap between
# them, we're facing a problem of overfitting. To face an overfitting problem, let's
# increase the complexity of the dataset.
polyfeats = PolynomialFeatures(degree=2)
X_train_poly = polyfeats.fit_transform(X_train)
X_test_poly = polyfeats.transform(X_test)

# After applying polynomial features to the train and test set, our train and test
# set spiked up in complexity: they now are: train set (354,105) and test set (152,105).
# They present 92 column more (a lot of complexity added) than the original dataset.
X_train_poly.shape
X_test_poly.shape

# Before executing the regression, let's standardize the train and test datatasets
ss = StandardScaler()
X_train_poly = ss.fit_transform(X_train_poly)
X_test_poly = ss.transform(X_test_poly)

# Let's execute our linear polynomial regression
polylr = LinearRegression()
polylr.fit(X_train_poly, Y_train)


# Basically, to recognize the overfitting, we need to train our model, and make him
# be able to do prediction. Once he can predict, we're going to predict targets both
# on the test set and the train set. Once done that, we mesure the error with the aid
# of the goodness metrics to establish if we got an overfitting problem: if the error
# on both the prediction done on the test and the train is DISTANT between them, we got
# an overfitting problem.
# Let's then predict the targets using the test set and the train set (polynomial).
Y_pred = polylr.predict(X_test_poly)
Y_pred_train = polylr.predict(X_train_poly)

# Let's now calculate the R2 score and MSE of both the prediction
r2_score(Y_test, Y_pred)
r2_score(Y_train, Y_pred_train)
mean_squared_error(Y_test, Y_pred)
mean_squared_error(Y_train, Y_pred_train)

# Since both the R2 Score and the MSE done on both the prediction from the test and
# the train set presents an huge gap (R2 Pred Test: 0.69 - R2 Pred Train: 0.89 and
# MSE Pred Test: 28.25 MSE Pred Train: 8.22) this built model present an overfitting problem.

# NOTE: BEFORE DOING REGOLARIZATION, DATA MUST BE ON THE SAME SCALE: USE NORMALIZATION
# OR STANDARDIZATION TO ACHIEVE THAT.

# Now, let's try to apply a regolarization to our model to remove the overfitting.
# We got two kind of regolarization:
# L2 (Or weight decay) will REDUCE the magnitude of the bigger weight used for the model.
# L1 will REDUCE to 0 the LEAST IMPORTANT weight used for the model.
# L1 and L2 regolarization are contained into some models that sklearn offer us.
# Let's start with the L2 regolarization: We'll import the Ridge model for that.
from sklearn.linear_model import Ridge

# Let's now define the LAMBDAS (the hyperparameter linked to the weight that the
# regolarization will have inside the model learning): we know that, lambdas should
# be contained in that range: 10^-4 < lambda < 10. Let's then define some values
# between those. NOTE: sklearn refer to lambda as "alpha".

alphas = [0.001, 0.001, 0.1, 1., 5., 10.]

# let's now build our model. Since we want to test all the possible alphas (lambda)
# let's declare and train the model into a for loop.
# At the time of this test was made, the best attemp is with Alpha = 0.1:
# Without L2, R2 Score for test was 0.69 and 0.89 for train.
# With L2 and an alpha of 0.1, the R2 Score test is 0.88 for test and 0.92 for train:
# As we can see, we almost removed entirely the problem of the overfitting.

# for each possible alpha into the array
for alpha in alphas:

    # assign the model to a variable: we pass the current alpha (lambda) to the model
    # to set the right hyperparameter
    ridgel2 = Ridge(alpha=alpha)

    # train the model with the train sets
    ridgel2.fit(X_train_poly, Y_train)

    # let's now predict both the train and the test set with predit; we'll use them
    # to analyze the gap between the errors to check if the overfitting was improved
    Y_ridge_pred = ridgel2.predict(X_test_poly)
    Y_ridge_pred_train = ridgel2.predict(X_train_poly)

    # Let's now check the errors: R2 score and the MSE for both the sets, and let's
    # then compare between them
    r2_test_ridge = r2_score(Y_pred, Y_ridge_pred)
    r2_train_ridge = r2_score(Y_train, Y_ridge_pred_train)
    mse_test_ridge = mean_squared_error(Y_pred, Y_ridge_pred)
    mse_train_ridge = mean_squared_error(Y_train, Y_ridge_pred_train)

    # print the info
    print("Lamda={} - R2 Test: {} - R2 Train {} / MSE Test: {} - MSE Train: {}".format(
    alpha, r2_test_ridge, r2_train_ridge, mse_test_ridge, mse_train_ridge
    ))

# Now, we'll do the same with another model, containing the L1 regolarization.
# This model is contained into the sklearn modules, and it's called Lasso.
# Let's build the model and train it the exact same way as we did above.
# Test result: As we know, L1 lower to 0 least important weight, so it kinda "remove"
# some complexity from the dataset. Let's compare results:
# Without L2, R2 Score for test was 0.69 and 0.89 for train.
# With L1 and an alpha of 0.1, the R2 Score test is 0.820 for test and 0.828 for train:
# We REMOVED ENTIRELY the overfitting problem, but we also lowered a little the accuracy of
# our model. We can also see that with alpha = 10 the model is literally broke, because with
# an high lambda, all the weights are lowered to 0.
from sklearn.linear_model import Lasso

for alpha in alphas:

    # assign the model
    lassol1 = Lasso(alpha=alpha)

    # train the model
    lassol1.fit(X_train_poly, Y_train)

    # predict both train and test to compare the errors later
    Y_lasso_pred = lassol1.predict(X_test_poly)
    Y_lasso_pred_train = lassol1.predict(X_train_poly)

    # Check scores
    r2_test_lasso = r2_score(Y_test, Y_lasso_pred)
    r2_train_lasso = r2_score(Y_train, Y_lasso_pred_train)
    mse_test_lasso = r2_score(Y_test, Y_lasso_pred)
    mse_train_lasso = r2_score(Y_train, Y_lasso_pred_train)

    # Print errors
    print("Lamda={} - R2 Test: {} - R2 Train {} / MSE Test: {} - MSE Train: {}".format(
    alpha, r2_test_lasso, r2_train_lasso, mse_test_lasso, mse_train_lasso
    ))


# Last model is something that uses at the SAME TIME both L1 and L2 regolarization.
# We import that from sklearn using the ElasticNet model. We will act the same as above,
# putting the initialization in a for cycle and using our alphas.
# Let's build the model and train it the exact same way as we did above.
# Let's then compare results:
# Without L2/L1 Elastic net, R2 Score for test was 0.69 and 0.89 for train.
# With L2/L1 Elastic net and an alpha of 0.1, the R2 Score test is 0.821 for test and 0.825 for train
# And with alpha = 0.001 R2 was 0.86 for test and 0.90 for train.
from sklearn.linear_model import ElasticNet

for alpha in alphas:

    # Declaring the elasticnet model, we need another argument: the l1_ratio.
    # This will affect the % of which L2 and L1 are present inside the model.
    # Putting l1_ratio to 0.5 will assure that L1 and L2 will be EQUALLY used.
    en = ElasticNet(alpha=alpha, l1_ratio=0.5)

    # let's train our model
    en.fit(X_train_poly, Y_train)

    # Then predict
    Y_pred_en = en.predict(X_test_poly)
    Y_train_en = en.predict(X_train_poly)

    # Check scores
    r2_test_en = r2_score(Y_test, Y_pred_en)
    r2_train_en = r2_score(Y_train, Y_train_en)
    mse_test_en = mean_squared_error(Y_test, Y_pred_en)
    mse_train_en = mean_squared_error(Y_train, Y_train_en)

    # Print errors
    print("Lamda={} - R2 Test: {} - R2 Train {} / MSE Test: {} - MSE Train: {}".format(
    alpha, r2_test_en, r2_train_en, mse_test_en, mse_train_en
    ))
