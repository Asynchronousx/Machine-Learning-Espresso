import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

# Import our dataset through load_digits
digits = load_digits()

# Assign properties and target to two numpy arrays
X = digits.data
Y = digits.target

# Split train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=420)

# Since we're going to use a NON-LINEAR model, let's normalize data instead of standardizing it
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

# Let's build our actual KNN model: The K-NN is a simple and efficent NON-LINEAR Classification model,
# that rely his classification on a simple idea: NN stays for Nearest Neighbor, and K stays for the K
# Nearest Neighbor from our example to classify. The algorithm will create a sort of "circle" of radius K
# that will embrace the K nearest object from all the present classes in this circle. The algorithm build this
# circle calculating the distance between the new example to classify and the class member around him with a
# specific metric: Euclidean distance, Manhattan distance or Minkwowski distance (NB: Standard for sklearn KNN is
# Minkwowski). Once done that, the algorithm will proceede to check the major number of classes object inside this ratio:
# If we got 5 elements, of which 3 belong to a square class, and 2 to a triangle class, the new example will be classified as
# a square. Also, we can have a draw: if we got 2 square, 2 triangle and 1 star, we'll mesure the distance between the new
# example and each object of the class that got a draw: the lesser distance win, and the example will be classified.
# Let's build our model, train and predict as we know
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

# train
knn = knn.fit(X_train, Y_train)

# predict probability and predictions for the test set
Y_pred = knn.predict(X_test)
Y_pred_prob = knn.predict_proba(X_test)

# do that also for the train set; we're going to check if there is overfitting
Y_pred_train = knn.predict(X_train)
Y_pred_prob_train = knn.predict_proba(X_train)

# Compare and show the metric
print("TEST ACCURACY = {} - TRAIN ACCURACY = {}\nTEST LOG LOSS = {} - TRAIN LOG LOSS = {}".format(
      accuracy_score(Y_test, Y_pred),
      accuracy_score(Y_train, Y_pred_train),
      log_loss(Y_test, Y_pred_prob),
      log_loss(Y_train, Y_pred_prob_train)
))

# The accuracy seems pretty high, but what concern us is the log loss. For the test set we can see
# that the log loss is 0.28, while for the train set 0.02. And that's a lot.
# We now want to check then, with DIFFERENT K as neighbors, what would happen. Lets then do that specifying
# our custom vecotr of Ks, and defining the model in a for loop using those K.
Ks = [1,2,3,4,5,7,10,15,20]

for K in Ks:
    print("Actual K: {}".format(K))

    # build and train the model with the Kth element
    knn_kth = KNeighborsClassifier(n_neighbors=K)
    knn_kth.fit(X_train, Y_train)

    # predict both predictions and probability, for test and train set
    Ykth_pred = knn_kth.predict(X_test)
    Ykth_pred_prob = knn_kth.predict_proba(X_test)
    Ykth_pred_train = knn_kth.predict(X_train)
    Ykth_pred_prob_train = knn_kth.predict_proba(X_train)

    # show infos: We achieve the best performance around K=10/15.
    print("TEST ACCURACY = {} - TRAIN ACCURACY = {}\nTEST LOG LOSS = {} - TRAIN LOG LOSS = {}\n".format(
          accuracy_score(Y_test, Ykth_pred),
          accuracy_score(Y_train, Ykth_pred_train),
          log_loss(Y_test, Ykth_pred_prob),
          log_loss(Y_train, Ykth_pred_prob_train)
    ))

# Let's now show where the KNN failed to predict the correct numbers.
# We'll then use an heatmap to achieve that
cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm,
            cmap="Blues_r",
            linewidths=.5,
            annot=True
)
plt.xlabel("Predictions")
plt.ylabel("Correct Values")
