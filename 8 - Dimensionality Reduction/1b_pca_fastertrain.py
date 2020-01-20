import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from time import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

# chdir to current then import the load mnist function
os.chdir("/path/to/mnist")
from mnistloader import load_mnist

# As done in the MultilayerPerceptron example, let's load the MNIST dataset through the function
# load_mnist, contained into the mnistloader.py file. The function will return four elements; the XY train and test set.
X_train, X_test, Y_train, Y_test = load_mnist(path="mnist")

# Take data on the same scale with normalization (since we're going to use LogisticRegression)
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test  = mms.transform(X_test)

# Check the properties number of the train/test set: as we can see, there are 784 PROPERTIES, because the images are in form of
# 28x28 pixel matrices.
X_train[0].reshape(28,28).shape
X_train.shape

# Let's now make a test: before applying PCA on the dataset, let's calculate how much time we need to train the LogisticRegression model
# without the dimension reduction. We'll use the function time to achieve that
logreg = LogisticRegression(max_iter=1000)
start = time()
logreg.fit(X_train, Y_train)
end = time()

# displaying time elapsed
elapsed = end - start
elapsed

# predicting both actual prediction and probability
Y_pred = logreg.predict(X_test)
Y_pred_train = logreg.predict(X_train)
Y_pred_prob = logreg.predict_proba(X_test)
Y_pred_train_prob = logreg.predict_proba(X_train)

# checking accuracy and log loss on the trained model
print("ACCURACY - TEST = {} / TRAIN = {}\nLOG LOSS - TEST = {} / TRAIN = {}".format(
      accuracy_score(Y_test, Y_pred),
      accuracy_score(Y_train, Y_pred_train),
      log_loss(Y_test, Y_pred_prob),
      log_loss(Y_train, Y_pred_train_prob)
))

# plotting the confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm,
            cmap='Blues_r',
            annot=True,
            annot_kws={"size":12},
            linewidths=.5
)

# Now, let's use PCA to shrink dimensionality. We'll pass as argument a variance trheshold of 0.90, in which only the values
# passing this costraint will be considered.
pca = PCA(0.90)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Check the reduced shape, if not satisfying go back and change the variance trheshold: Now properties should have gone down from
# 754 to 87
X_train_pca.shape
X_test_pca.shape

# Train the model again; let's create another one
logreg_pca = LogisticRegression(max_iter=1000)
start_pca = time()
logreg_pca.fit(X_train_pca, Y_train)
end_pca = time()

elapsed_pca = end_pca - start_pca
elapsed_pca

# let's do predictions
Y_pred_pca = logreg_pca.predict(X_test_pca)
Y_pred_train_pca = logreg_pca.predict(X_train_pca)
Y_pred_prob_pca = logreg_pca.predict_proba(X_test_pca)
Y_pred_train_prob_pca = logreg_pca.predict_proba(X_train_pca)

# checking accuracy and log loss on the trained model
print("ACCURACY - TEST = {} / TRAIN = {}\nLOG LOSS - TEST = {} / TRAIN = {}".format(
      accuracy_score(Y_test, Y_pred_pca),
      accuracy_score(Y_train, Y_pred_train_pca),
      log_loss(Y_test, Y_pred_prob_pca),
      log_loss(Y_train, Y_pred_train_prob_pca)
))
y_p
a = np.array([accuracy_score(Y_test, Y_pred), accuracy_score(Y_train, Y_pred_train)])
apca = np.array([accuracy_score(Y_test, Y_pred_pca),accuracy_score(Y_train, Y_pred_train_pca)])

plt.bar(a.mean(), elapsed)
plt.bar(apca.mean(), elapsed_pca)
plt.xlabel("Accuracy (larger the better)")
plt.ylabel("Time Elapsed")
plt.savefig("banana.png")

# We then got:
# WITHOUT PCA:
# TIME ELAPSED -> 556 SECONDS
# ACCURACY - TEST = 0.9256 / TRAIN = 0.9393833333333333
#LOG LOSS - TEST = 0.2712860299594799 / TRAIN = 0.22072140157270184

# WITH PCA
# TIME ELAPSED -> 16 SECONDS
# ACCURACY - TEST = 0.9195 / TRAIN = 0.9201833333333334
# LOG LOSS - TEST = 0.2825788146649393 / TRAIN = 0.2841295195590419
