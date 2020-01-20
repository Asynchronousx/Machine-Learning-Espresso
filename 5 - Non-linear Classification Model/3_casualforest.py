import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Loading the titanic dataset through pandas
titanic = pd.read_csv("http://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")

# check infos about the dataset
titanic.info()

# check the first entries of the dataframe
titanic.head()

# Let's drop the unusued column; we do not need the name one.
titanic = titanic.drop("Name", axis=1)

# And, since we got SEX represent as a categorical nominal property, let's do the one hot encoding on that
titanic = pd.get_dummies(titanic)

# Now, what is a casual forest? A casual forest is a ML model that uses the ENSEMBLE learning.
# We could describe ENSEMBLE learning as a procedure that follow one simple and solid principle: we build a model made by multiple models
# that result in a more robust and stable model than taking the single one.
# A casual forest is then made by MULTIPLE decision tree, in which, every of them will made predictions: at the end of the training and
# prediction phase, we'll merge all the prediction in a single one using the mean: the result will be usually much better than a single
# decision tree. The unique downside is that is compunationally heavy.
# Let's build the numpy dataset and split the dataset into train and test
X = titanic.drop("Survived", axis=1).values
Y = titanic["Survived"].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=420)

# Now let's build our actual model
from sklearn.ensemble import RandomForestClassifier

# Here, we're building a random forest classifier utilizing Gini as IMPURITY metric calculator, setting random state to false to having
# consistent results on multiple train, the max depth of the tree setted to 8 and 30 decision tree into this forest.
rfc = RandomForestClassifier(criterion="gini", random_state=False, max_depth=7, n_estimators=30)

# Let's train the model
rfc.fit(X_train, Y_train)

# And then predict both on test and train set to compare the accuracy metric
Y_pred = rfc.predict(X_test)
Y_pred_train = rfc.predict(X_train)

# Print the metrics and check for overfitting; if results are not any good, reduce deepness and increase the estimators.
print("RFC - TEST ACCURACY = {} / TRAIN ACCURACY = {}".format(
      accuracy_score(Y_test, Y_pred),
      accuracy_score(Y_train, Y_pred_train)
))
