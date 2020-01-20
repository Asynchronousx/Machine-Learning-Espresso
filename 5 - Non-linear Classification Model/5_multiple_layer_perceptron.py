import os
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss


# We're going to create a MLP neural network. Since NeuralNetworks NEEDS A LOT OF EXAMPLES to be
# reliable and robust, we should need a dataset with an huge number of entries. The MNIST dataset is the
# best option to use out there, to test our MLPNN. We got two ways to do that:
# 1) Going diretly to the MNIST site, download their 4 zip (Train set & label, Test set & label) and build
# the set appending labels to both the set: this is a fun solution but require an apposit function to assign
# labels to dataset.
# 2) Using tensorflow, and using the mnist dataset contained into the keras module. Let's choose that.
# Let's use a custom function defined into the file mnistloader.py: load_mnist.

# first step: assure to be in the right directory. If not, change the directory to the desired one using the
# os chdir func; be sure to chdir into the folder in which the "mnist" folder is located
os.getcwd()
os.chdir("/path/to/mnist")

# then import the module
from mnistloader import load_mnist

# Let's now load the data from the dataset: calling the function load_mnist, we pass as argument the path
# where the 4 files are stored (if current path, just leave it blank). Also, be sure to extract the files.
X_train, X_test, Y_train, Y_test = load_mnist(path="mnist")

# Once loaded, let's do the normalization on the datasets to scale the data.
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

# Now, before using a neural network, good practice is to check that the problem can or can't be solved by a simpler
# model, for example, for our classification, a logistic regression. Let's then build, train and predict the results
# using a logistic regression model, calculating the accuracy.
logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train, Y_train)

# do prediction, both for test and train set; calculate also the probability
Y_pred = logreg.predict(X_test)
Y_pred_train = logreg.predict(X_train)
Y_pred_prob = logreg.predict_proba(X_test)
Y_pred_train_prob = logreg.predict_proba(X_train)

# Print accuracy using accuracy_score and log_loss: The results with LogisticRegression are quite good,
# showing for 200 iteration a scores of:
# ACCURACY: TEST = 0.9261 / TRAIN = 0.9384833333333333
# LOG LOSS: TEST = 0.09466340765112355 / TRAIN = 0.22463870717178308
print("ACCURACY: TEST = {} / TRAIN = {}\nLOG LOSS: TEST = {} / TRAIN = {}".format(
      accuracy_score(Y_test, Y_pred),
      accuracy_score(Y_train, Y_pred_train),
      log_loss(Y_pred, Y_pred_prob),
      log_loss(Y_train, Y_pred_train_prob)
))

# For the knowledge sake, let's now build our MLP neural network. Basically, a MLP is an ENSEMBLE model
# where a PERCEPTRON (or Artificial Neuron) is the single model. We remember that an artificial neuron is
# composed by a core, in which are defined the Signal Potential formula (Usually in form of Z = b + w1x1 + ... + wnxn)
# and an activation function, let's call that G that will take in input Z and returns the output: since we usually use
# ReLU as activation function for the hidden layers (that are layers of N perceptron that will do the predictions) it will
# return the computation of Z itself. For the output layer we usually use a SIGMOID activation function, since it will also return
# the probability of accuracy. this new Z value will be then the INPUT for the NEXT HIDDEN LAYER if present (becoming a property).
# So, with N properties, each of them will be mapped to a neuron with a weight w, and that's happens for all the neuron
# in the network. Each neuron will then compute the value Z through the signal potential and activation function, and will be used
# in the same exact way for the next hidder layer (acting then as a property). We increase the accuracy of the network with the
# backpropagation technique, that will assure the best accuracy possible with the lowest loss.
# NOTE: this is purely academical. Usually for better NN we use TF, Pytorch or Keras.
# Additional NOTE: Epoch -> iteration cycle composed by those steps: weight assignment, prediction, error measurement, backpropagation and
# calculation of new weight. A training for a NN ends when a trheshold is reached (no changes from an epoch to another) or when the max
# number of epochs are finished.
# LET'S Import the MLP module from sklearn
from sklearn.neural_network import MLPClassifier

# Let's build our actual network: we'll specify the number of neuron for each layer (N, M, O, .., Z). We can also build a SINGLE
# hidden layer MLP. we also pass verbose = true to check, for each epoch, the
mlp = MLPClassifier(hidden_layer_sizes=(100,), verbose=True)

# train the model: this steps can needs some minutes based on how much neurons and layer are specified.
mlp.fit(X_train, Y_train)

# Build predictions, both for examples and probabilities and then lets compare as before
Y_pred_mlp = mlp.predict(X_test)
Y_pred_train_mlp = mlp.predict(X_train)
Y_pred_prob_mlp = mlp.predict_proba(X_test)
Y_pred_train_prob_mlp = mlp.predict_proba(X_train)

# Showing the results:
print("ACCURACY: TEST = {} / TRAIN = {}\nLOG LOSS: TEST = {} / TRAIN = {}".format(
      accuracy_score(Y_test, Y_pred_mlp),
      accuracy_score(Y_train, Y_pred_train_mlp),
      log_loss(Y_pred, Y_pred_prob_mlp),
      log_loss(Y_train, Y_pred_train_prob_mlp)
))
