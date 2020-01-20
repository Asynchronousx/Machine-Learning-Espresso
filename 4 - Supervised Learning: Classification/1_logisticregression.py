import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Function to show the decision boundary of the classification
def showBounds(model, X, Y, labels=["Benign","Malicious"]):

    h = .02

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    X_m = X[Y==1]
    X_b = X[Y==0]
    plt.scatter(X_b[:, 0], X_b[:, 1], c="green",  edgecolor='white', label=labels[0])
    plt.scatter(X_m[:, 0], X_m[:, 1], c="red",  edgecolor='white', label=labels[1])
    plt.legend()




breast_cancer = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                           names=["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"])

breast_cancer.head()

# Fetching some info about the dataframe: we got 32 columns, of which 31 are properties
# and just one is the target we want to know about: diagnosis.
breast_cancer.info()

# Printing info about diagnosi column: as we can see, the Y target dataset contains only the diagnosis
# values: those values are in form of a char, M and B (Malicious and Benign).
# To check that, we access the diagnosis column and call the function "unique" on that,
# checking how many different values are present inside the column. As stated, we got only M and B.
breast_cancer["diagnosis"].unique()

# Let's start building a BINOMIAL dataframe for our logistic regression: we'd like to
# analyze only two feature, radius_se and concave points_worst, and for the y target set
# we'll use diagnosis
X = breast_cancer[["radius_se", "concave points_worst"]].values
Y = breast_cancer["diagnosis"].values

# Let's now build our test set and train set with the technique we know
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=420)

# As we stated, the target values are in form of a character. We should then ENCODE those
# values, to transform them into number. BUT, an interesting property of the logistic regression
# implemented by sklearn, is that the regression itself is able to classify also non-numeric target
# (as in our case). A good practice however, is to encode the non-numeric variable into numbers,
# both using label encoding or one hot encoding. Since our target property is composed just by two
# class (M,B) we can use label encoding to specify that 1 means the memebership of a target into the
# positive class (1), meanwhile 0 means that is not a member of the positive class (negative class).
# Let's create a label encoder object and let's encode the label both for Y_TEST AND TRAIN.
# Since the encoding is not done in place, we must assign the result to the train and test set.
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)

# Once encoded our label, another good practice is to check if the data of the train/test set
# are on the same scale. Let's then standardize it. Since, like LabelEncoder the standardizer
# do not work in place, we need to assign it to the x train and test set
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# Once our data is preprocessed (encoded and standardized) we now need to create our LogisticRegression
# model. To do that, we simply use the LogisticRegression module from sklearn, that is the same as a
# LinearRegression model in every aspect.
from sklearn.linear_model import LogisticRegression

# Here, the peculiarity of the logistic regression, is that by standard, it implements the Regolarization
# of the data. More specifically, it implements the L2 Regolarization (weight decay) using a standard value
# of lambda (or alpha) of 1/lambda. The default lambda value is 1, because 1/1 = 1. Using an higher lambda results
# in a lessed impacted L2 regolarization (1/big number = small number) while using a small lambda results into an
# higher impacted L2 regolarization (1/small number = big number).
# the function, specifying the parameter, is so called: LogisticRegression(penalty="l2", C=1) where penalty is the
# type of regolarization (penalty on the high weight or lower weight) and C is the inverse of lambda (1/lambda)
# Let's build the default LogisticRegression.
logreg = LogisticRegression()

# let's now train our model as we already know from every regression model
logreg.fit(X_train, Y_train)

# With our model trained, we're going to build our prediction set as we already know.
# But, for the logistic regression, we use the standard prediction (made with predict) and also another prediction
# known as "predict_proba()". This method will return the values about how correct the prediction is in terms
# of probability.
Y_pred = logreg.predict(X_test)
Y_pred_prob = logreg.predict_proba(X_test)

# Now, for evaluating how good our predictions are, we're not going to use R2 score and MSE because this is not a linear
# model. We're going to use two new feature:
# accuracy_score -> percentual of the correct predictions
# log likelyhood -> probability that quantify how much the prediction done is correct: Since sklearn implements the negative
# of the log likelyhood, a lower value of this metric means an higher quality of the model.
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

# let's now print those values: accuracy score works just like R2 and takes in input the test and prediction target set.
# the log loss takes in input the test set and the probability dataset calculated with predict_proba, since Y_pred_prob
# contains, for each test record, the probability on "how the target value will be correct". log loss will then compare
# each probability for each Y_test value and return the goodness of the predictions.
print("ACC_SCORE = {} - LOG LIKELYHOOD (LOG LOSS) = {}".format(
    accuracy_score(Y_test, Y_pred),
    log_loss(Y_test, Y_pred_prob)
))

# Let's now show the decision boundary created by the predictions. The decision boundary is nonentheless the line that
# separe the classes in two areas: this boundary represent then the membership of a value to a class instead of another.
showBounds(logreg, X_train, Y_train)

# Show the decision boundary on the test set
showBounds(logreg, X_test, Y_test)

# As we can see from those graph, the decision of the class membership is quite good.
# But, we can even enhance those results by including ALL THE PROPERTIES into the train and test set,
# since more data imply more accuracy. Let's then build our new sets with those properties and follow the
# exact same step as before. (NOTE: Xe and Ye means X entire and Y entire, for including the entire properties into the set.)
Xe = breast_cancer.drop(["id", "diagnosis"], axis=1).values
Ye = breast_cancer["diagnosis"].values

# splitting datasets
Xe_train, Xe_test, Ye_train, Ye_test = train_test_split(Xe, Ye, test_size=0.3, random_state=420)

# Encoding the labels in the target set
Ye_train = le.fit_transform(Ye_train)
Ye_test = le.transform(Ye_test)

# Standardize data (necessary to scale data on all those properties (more then 30))
Xe_train = ss.fit_transform(Xe_train)
Xe_test = ss.transform(Xe_test)

# Build the LogisticRegression model logreg_e = LogisticRegression entire
logreg_e= LogisticRegression()
logreg_e.fit(Xe_train, Ye_train)

# predict both the result set and the probability set
Ye_pred = logreg_e.predict(Xe_test)
Ye_pred_prob = logreg_e.predict_proba(Xe_test)

# calculate the error with accuracy_score and log_loss: as we can state, the accuracy score is now even higher
# because we used all the data on the dataset. Also, the log loss is reduced (lesser value = better probability
# of predictin being correct)
print("ACC_SCORE = {} - LOG LIKELYHOOD (LOG LOSS) = {}".format(
      accuracy_score(Ye_test, Ye_pred),
      log_loss(Ye_test, Ye_pred_prob)
))

# Build the decision boundary graph to show the goodness of the prediction and classification both on train and test
# train
showBounds(logreg, X_train, Y_train)

# test
showBounds(logreg, X_test, Y_test)
