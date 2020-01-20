import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# For this code, we'll use the digits dataset contained inside the dataset module of
# sklearn
from sklearn.datasets import load_digits

# since load_digits is a method, we need to istantiate it into a variable
digits = load_digits()

# the type of digits is neither a NUMPY array nor a dataframe, but a special type of sklearn
# datasets. No problem: we can still get numpy arrays by utilizing data and target function
# of the dataset, that will return properties and target of the set.
type(digits)

# now we got a numpy array
X = digits.data
Y = digits.target

Y

# let's see the structure of the X and Y sets: since we loaded the digit dataset, the resultant
# set got 1797 examples and 64 columns. Why 64 columns? Because the digit dataset contains images
# in form of pixels: each image got a size of 8x8, pretty low quality, and then it got 64 pixels in
# total. Thats why we got 64 columns, because each of them contains a pixel value for a given (row,column).
# The Y target set instead, contains only 1797 records, and each of this records is, for the given row in the
# X dataset, the number it represents.
X.shape
Y.shape

# Let's check the unique value inside this array: they should be digits in range 0-9.
np.unique(Y).shape

# And now, let's try to visualize each of those digits as an image. Since we got 10 digits (0-9) we're going
# to iterate 10 times to get them all. To achieve that we can use the unique element of the Y set (the targets).
# they're obviously 0-9.
 for digit in np.unique(Y):

    # We now need to fetch the actual 64-column vector from the X dataset (since it contains 1797 records composed by
    # 64 columns, in which each columns represent a pixel). We'd like to return, for showing porpuses, only the vectors
    # containing the i-th class (the digit 0, the digit 1, the digit 2... the digit 9).
    # To achieve that we apply a mask on the Y array target set: writing Y==digit (0,1,2...9) will return, each iteration,
    # a mask containing 1797 record where each record will be True or False (a mask): the True record means that the value
    # at that row MATCH the digit variable passed as the mask costraint: applying that True/False array on the X numpy properties set
    # will the return ONLY the row that match the True value.
    # I.E:
    # X = np.mat([[1,2], [3,4], [5,6], [7,8], [9,10], [11,12]])
    # Y = np.array([0,1,2,8,9,1])
    # doing Y==1 will return the array [false, true, false, false, false, true]. Then, doing X[Y==1] will return an array composed
    # only by the rows that match the true value: [[3,4], [11,12]]. Doing then X[0] will return only [3,4].
    # We're taking the first value [X[0]] of this masked array just to print the specific digit actually analyzed.
    digit_image = X[Y==digit]

    # Assign to digit image the first entry of the new returner array containing ALL the digit matrixes, specifying that we want
    # only the first one digit_image[0] and reshaping it to a 8x8 matrix.
    digit_image = digit_image[0].reshape([8,8])

    # use imshow to build up the figure: we pass image and the colormap as argument
    plt.imshow(digit_image, cmap="gray", )

    # show the image
    plt.show()

# After showed the digits (not necessary; for the sake of curiosity) we can now start to preprocess the data. Since those are matrixes
# composed by pixel values in each column, those value floats between 0 and 255; That because the images are built by using grayscale
# values. To achieve better train performance, normalize those images betwen 0 and 1: we use MinMaxScaler to achieve that.
mms = MinMaxScaler()


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=420)

# Normalize data
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

# Let's execute our multiclass classification: we're going to do nothing more than a simple logistic regression. Luckily for us, the
# sklearn LogisticRegression model, CAN RECOGNIZE BY ITSELF when we're into a multiclass classification problem. He will then adopt
# automatically the OneVSAll classification (that, we remember, having N classes, we apply the logistic regression to each of them,
# one by one, doing for each class a binomial classification: for N classes, 1 is threated as the positive class (1) and the other N-1
# as the negative one (0). We do that for each class, resulting in N classifiers that will, with a new occurrence to test, return the
# probability of which class the new occurrence belongs).
# Let's simply build a logistic regression model as we know: in this case, 100 iteration (standard) are not enough to fit the model.
# the least number of iteration possible are 164.
logreg = LogisticRegression(max_iter=164)
logreg.fit(X_train, Y_train)

# Let's predict both the prediction set containing the result, and the probability that each result have to be a correct prediction
Y_pred = logreg.predict(X_test)
Y_pred_proba = logreg.predict_proba(X_test)

# Let's know print the metrics (accuracy_score and log_loss to check how good the predictions are)
print("ACCURACY_SCORE = {} / LOG LIKELYHOOD (LOG LOSS) = {}".format(
      accuracy_score(Y_test, Y_pred),
      log_loss(Y_test, Y_pred_proba)
))

# Let's introduce a new metric for the accuracy testing of our model, the confusion matrix.
from sklearn.metrics import confusion_matrix

# Building the matrix passing in input the test set and the prediction set to make comparison
cm = confusion_matrix(Y_test, Y_pred)

# Showind the matrix: what are those values? The CM is a matrix composed by an NxN size, where N is the number of the total class
# of the dataset. In each element [row][column] we do have the numbers of total prediction for that row (or class):
# I.E: Row 0 represent, in our case, the digit 0. Each column represent the value we predicted training our model and using predict
# on the test set. for example, having, at the row 0 the values [48,  0,  0,  0,  0,  0,  0,  0,  0,  0] means that the model recognized
# ALL the 0 values as 0 (because only the 0th column contains a value different than 0).
# For another example, the 1th row (the value 1) is presented like [ 0, 56,  0,  1,  0,  0,  0,  0,  1,  1]: that mean that our model
# was able to regonize 56 number 1 correctly, a number one as a 3 (4th column = 1) and a number one as an eight and another as a nine.
# and so on. It represent then the correctness of the predictions in base on how many results are correct.
print(cm)

# To achieve a better visualization, let's use seaborn and the heatmaps
import seaborn as sns

# Note that, SNS uses matplotlib to plot figures. Since we're going to whitespace elements to have a better understading, to avoid
# plotting a text separated from the whitespace we can operate directly on the image we're going to plot using plt and modifying the
# attributes of the next figure (mantained in memory by sns) using the matplotlib function.
sns.heatmap(cm,
            cmap="Blues_r",
            linewidths=.5,
            annot=True
)
plt.xlabel("Predict Class")
plt.ylabel("Correct Class")

# NOTE: To achieve multiclass classifcation, sklearn offers a PRECISE model to do that. This model is called AllVSRest.

# The peculiarity of the OneVsRestClassifier is that he need, in the instantiation process, an actual classifier model to be assigned
# into his internal structure. We can easily pass the LogisticRegression classifier since the OneVSAll concept rely on the logistic
# regression applied to the N classes of the dataset.
ovr = OneVsRestClassifier(LogisticRegression())

# Create new test and train set
Xmulti_train, Xmulti_test, Ymulti_train, Ymulti_test = train_test_split(X,Y, test_size=0.3, random_state=420)

# normalize
mss = MinMaxScaler()
Xmulti_train = mss.fit_transform(Xmulti_train)
Xmulti_test = mss.transform(Xmulti_test)

# Let's now use this classifier as we use them all: fit, predict and test accuracy
ovr.fit(Xmulti_train, Ymulti_train)
Ymulti_pred = ovr.predict(Xmulti_test)
Ymulti_pred_proba = ovr.predict_proba(Xmulti_test)

# check accuracy: spoiler - the results are EXACTLY THE SAME as we did above: cause the OneVsRestClassifier is just a kinda of
# interface, but the real work is done by the classifier passed in input.
print("ACCURACY_SCORE = {} / LOG LIKELYHOOD (LOG LOSS) = {}".format(
      accuracy_score(Ymulti_test, Ymulti_pred),
      log_loss(Ymulti_test, Ymulti_pred_proba)
))
