import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Function to show bounds
def plot_bounds(X,Y,model=None,classes=None, figsize=(8,6)):

    plt.figure(figsize=figsize)

    if(model):
        X_plot_train, X_plot_test = X
        Y_plot_train, Y_plot_test = Y
        X = np.vstack([X_plot_train, X_plot_test])
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

        xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                             np.arange(y_min, y_max, .02))

        if hasattr(model, "predict_proba"):
            Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        else:
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=.8)

    plt.scatter(X_train[:,0], X_train[:,1], c=Y_train)
    plt.scatter(X_test[:,0], X_test[:,1], c=Y_test, alpha=0.6)

    plt.show()

# Load the iris dataset directly from the uci.edu site
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                   names=["sepal length","sepal width","petal length","petal width","class"])

# displaying first elements
iris.head(100)

# show how many values are unique into the target class
iris["class"].unique()

# As we can see, the target is represented by "Class", which hold the name of the plant. Since we simply want,
# given a set of data, to assign the new example to one of those class, we do not need the names anymore: we can just
# encode the class column to be represented by number. You can ask: since Class is a column represented by a categorical
# nominal value, why don't we use the One Hot Encoding? Because, that would be correct if the class wouldn't be a target one:
# if "Class" was a property of our set, we should have needed to use one hot encoding to take track of all the class (Setosa,
# versicolor etc.) but since the column "Class" is a target, we just need to classify each of them as a number. And for that,
# we use label encoder. Before applying the label encoder we must extract the dataset in form of numpy array.
X = iris.drop("class",axis=1).values
Y = iris["class"].values

# split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=420)

# Let's encode the Y train and test set (class column)
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)

# Standardize data inside the X train and test vector; we're now standardizing cause the SVM is a linear model. And linear model
# do prefer standardization instead of normalization.
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# Let's build our actual model: SVM (Support Vector Machine). The SVM is a particoular kind of linear classifier model, that have
# an aim of "maximizing" the margin between the classes instead of "drawing a line to separe them". We proceed to take a SMALL subset
# of the dataset, and train the model to this subset: generally, a SVM takes the most difficult example in this small parts, because
# it follow a simple principle: Done the classfication on the hardest cases, the less complex one will be easier".
# So, for example, the main difference from a logistic regression (that consider ALL the dataset and take the MOST REPRESENTATIVE
# example for a dataset for the training) is pretty much this.
# Those few example taken are called "Vector Support", because they're the only examples that are useful to the model creation.
# Take a small fraction of the dataset
X_train_small = X_train[:, :2]
X_test_small = X_test[:, :2]

# Let's import the SVM and build the model
from sklearn.svm import LinearSVC

svc = LinearSVC()

# Note: to reduce overfitting, we can specify the type of regolarization (L2 (Ridge), L1 (Lasso) or mix of both (ElasticNet))
# and also try to set the hyperparameter C, that we remember is the INVERSE of lambda (C big -> alpha small, C small -> alpha big)
svc.fit(X_train_small, Y_train)

# Now, instead of training the model, we can use directly one of his function, score: svc.score() will predict and compare the
# metric in just one function, useful for testing purpose as our.
print("TEST ACCURACY - {} / TRAIN ACCURACY = {}".format(
      svc.score(X_test_small, Y_test),
      svc.score(X_train_small, Y_train)
))

# PLOTTING: as we can see, since the SVM is a linear classificator, the boundary will be linear aswell.
# We note that, the SVM maximized the margin between the support machine (the closest dot to the boundary) and the
# linear boundary plotted, gaving us a good accuracy on the prediction.
plot_bounds((X_train_small, X_test_small), (Y_train, Y_test), svc, figsize=(8,6))

# Now, let's train on all properties of the set: create a new model and train it
§svc_e = LinearSVC()
svc_e.fit(X_train, Y_train)

# accuracy testing for the entire dataset properties using score as before
print("TEST ACCURACY - {} / TRAIN ACCURACY = {}".format(
      svc_e.score(X_test, Y_test),
      svc_e.score(X_train, Y_train)
))
