import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# Function to show bounds
def plot_bounds(X, Y, model=None, classes=None, figsize=(8,8)):

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


# Now, we're going to create a simple dataset distribuited into a concentric circle, to show how the KernelSVM
# works. To do that, we should need a specific dataset containing properties distribuited in this specific way:
# to avoid wasting our time searching for such dataset, we could just use the tool make_circles of sklearn to
# make a dataset appearing as a concentric circle. To understand better, let's build one and visualizing it.
# Since we're using sklearn, the function make_circles will return two numpy array; X and Y.
# We pass as argument the noise (that will put some properties of a class into another class radius, to make the
# dataset dirtier), the factor that will be used to distribuite values of the classes (a factor of 0.5 means that
# the two classes are perfectly balanced) and if wanted the random_state with a value != 0, to have consistent
# results over multiple run.
X, Y = make_circles(noise=0.2, factor=0.5, random_state=420)

Y[:20]


# Let's now visualize those dataset with the matplotlib scatter function: we pass the properties set as the whole input:
# we'll use all the row and the first column as the first property and all the row and the second column as the second one.
# We'll then use Y as the color separator: since Y contains values in range 0-1, the value with 0 will be displayed as a color,
# the value with 1 as another.
plt.scatter(X[:,0], X[:, 1], c=Y)

# Now we're ready to build our KernelSVM: to do that, import the dedicated library from sklearn
from sklearn.svm import SVC

# We remember that, a KernelSVM is just a "trick" applied to an SVM: since the SVM is a linear classificator, in such cases
# (like the non-linear dataset we built above) is necessary to apply a non-linear model to classify the properties as good as
# possible. Since with the SVM we could NEVER do a good classifcation on this kind of dataset, we use the KernelSVM to simulate
# an increment of DIMENSIONALITY: since this process is really computationally HEAVY, we use a workaround to increase dimensionality:
# we use a gaussian formula into the radial basis kernel to simulate a gauss bell: it will then place a landmark (usually in the center
# of the dataset) and then calculate the distance from each point of the dataset from that landmark: the distance between the positive
# class and the landmark will be approximately ONE, instead with the negative class will be 0. This is applied even to more complex
# problem, just adding weights and properties as a normal regression (i.e: b + w1k(x,l) + w2k(x,l) >= 0)
# The RBF kernel is in form of K(x,l) = e^(-Y*abs(x-l)^2) where Y = 1/2*std^2, and K is the kernel function, x the example and l the landmark.
# The KernelSVM is a good classifier (with the RBF method) when coming to dataset classification that should require more dimension to be 
# accurate.
# To accademic purpose, let's implement the SVC class with various kernel (Linear, RBF, tanh etc)

# VARIOUS IMPLEMENTATION OF THE KERNELSVM
# LINEAR:
# This implementation is EQUAL to implement a LinearSVC from the sklearn module: We pass "linear" as kernel argument, and with
# probability = true we're meaning that it will contains the probability score of the correctness of predictions. NOTE that this, on large
# dataset will casue to slow down a little.
# kernelsvm = SVC(kernel="linear", probability=True)
# RBF (Radial Basis Gaussian):
# This should be the default when NOT KNOWING which to use. Is the most accurate on many cases.
# kernelsvm = SVC(kernel="rbf", probability=True)
# SIGMOID AND POLY: use whenever RBF returns bad results.
# kernelsvm = SVC(kernel="sigmoid", probability=True)
# kernelsvm = SVC(kernel="poly", probability=True)

kernelsvm = SVC(kernel="rbf", probability=True)

# split data into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=420)

# train the model and fit/calculate accuracy using the function score, then visualize it with show_bounds
kernelsvm.fit(X_train, Y_train)
print("ACCURACY - TEST = {} / TRAIN = {}".format(
      kernelsvm.score(X_test, Y_test),
      kernelsvm.score(X_train, Y_train)
))

plot_bounds((X_train, X_test), (Y_train, Y_test), kernelsvm)
