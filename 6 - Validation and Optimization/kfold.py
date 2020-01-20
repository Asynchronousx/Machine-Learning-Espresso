import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the iris dataset through site
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                   names=["sepal length","sepal width","petal length","petal width","class"])

# Displaying first elements
iris.head()

# Getting the numpy dataset
X = iris.drop("class", axis=1).values
Y = iris["class"]

# split dataet into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=420)

# do data scaling with standardization to get better results
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# create a logistic regression model
logreg = LogisticRegression()

# Now, we need to do Cross Validation using K-Fold validation.
# In the first place, WHY doing validation? Because of DATA LEAKAGE.
# Let's do an example: We create a model and we train it with our training dataset. We then move on,
# and test the model just created with a test set. The accuracy results do not satisfy us, then we move
# back to the model creation and tune the hyperparameters. We train again, and test. The results do not
# satisfy us yet, so we go back and repeat those steps until the test set give a satisfying results.
# NOTHING MORE WRONG! because with that attitude, we just overfitted the model at train time to our dataset,
# so in inference time (time in which we test the model with unseen data) the model will sucks bad.
# This cause data leakage: creation of unexpected additional information in the training data, allowing
# a model or machine learning algorithm to make unrealistically good predictions. To avoid this kind
# of problem, we do use the Cross Validation, more specifically the K-Fold validation. This will SPLIT
# our dataset in K fold partition, in which ONE of them will be used for testing, and the rest (N-1) for
# training. This will occurr for K times, until every single subset hasn't been used as a test subset.
# I.E.
# K = 3, F represent a Fold
#  |F1| -> Test set
#  |F2| -> Train set -> Train and test the used model on this disposition
#  |F3| -> Train set
# Iter. 1
#  |F1| -> Train
#  |F2| -> Test set -> Train and test the used model on this disposition
#  |F3| -> Train set
# Iter. 2
# Etc. NOTE: each time we use another fold as test and train sets, we SHUFFLE data to prevent loops.
# Etc.. The result will be K different model, in which each of them have been trained and tested on
# different data, holding then a DIFFERENT ERROR of prediction. We then make a MEAN on that errors and
# check if that satisfy us. If so, we pass to actually TRAIN AGAIN OUR MODEL with the original test and
# train set, because the accuracy will be high even on unseen data. IF NOT, we're going back to the model
# creation and tune the hyperparameters. This will avoid OVERFITTING and DATA LEAKAGE, since we're validating
# our model on unseen data each time.
# Let's import the module from sklearn
from sklearn.model_selection import cross_val_score

# This function is all what we need: it will do EVERYTHING on the inside, including splitting into K fold,
# assigning test and train set at each iteration, shuffling and even training and memorizing scores.
# We need to pass the model we want to test on, the properties and target dataset we want to use for training
# and evaluating, and the number of folds, indicated by cv=K.
score = cross_val_score(logreg, X_train, Y_train, cv=10)

# This function returns a numpy array, containing for each model tested, his linked score. We can just use
# score.mean() to check if the mean error satisfy us
score.mean()

# IMPORTANT: AFTER A K FOLD VALIDATION, REMEMBER TO TRAIN AGAIN THE MODEL ON THE TRAIN SET.
# If so, then we need to TRAIN THE MODEL AGAIN on the train set, because we know that this will act good
# on unseen data, and also because the model was trained on the data we passed into the cross_val_score func.
logreg.fit(X_train, Y_train)

# from here, predict and use any error score/accuracy comparator needed
# PREDICT
# ERROR CHECKING
# ETC...
