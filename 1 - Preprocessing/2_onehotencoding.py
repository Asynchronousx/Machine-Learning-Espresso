import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

shirts = pd.read_csv("MLData/shirt.csv")

shirts.head(15)

# Creating a NUMPY array representing the dataset. We can achieve that by calling
# to numpy function of pandas from the dataframe itself, and storing the resul.
X = shirts.to_numpy()
X[:15]

# NOTE: for LabelEncoding, seeing Price as target, we could have acted in this way to apply the label
# encoding:
# X = shirts.drop("prezzo", axis=1).values
# Y = shirts["prezzo"].values
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)
# le = LabelEncoder()
# X_train = le.fit_transform(X_train)
# X_test = le.transform(X_test)
# The effects would have been equal as following; but in the following code we're treating
# X as a SINGLE array and not a splitted one 

# Since we have TWO categorial variables into this dataset:
# Size = Categorical Ordinal,  since we can order the size from the smallest (S) to the largest (X)
# Color = Categorical Nominal, since we cannot establish an ordinal relation for colors.

# We can operate the substitution both on the dataframe and the numeric numpy array.
# dataframe:

# defining a dict to map sizes to specific values
size_dict = {"S":0, "M":1, "L":2, "XL":3}

# substitution of the previous categorical nominal values with numbers.
# here, we're accessing the column size of the shirts dataframe, and swapping values
# with the shirts["taglia"] column of the dataframe mapped with the new values
# defined by dict. Everytime map will encounter a S, M, L, OR XL it will be
# swapped with the value of that specific key element.
# We basically use map to substitute the KEY with his value.

shirts["taglia"] =  shirts["taglia"].map(size_dict)
shirts.head()

# Now, we need to do the same for the numpy array: since we can't use panda, we
# need to define our proper vectorized function:
# a vectorized function is simply a function that take a vector in input and operate
# on the entire vector on an efficent way. It will involve some loops, but they're
# mostly done on a LOW LEVEL LANG as C.
# we define our custom function map for the numpy array as follow
# We use numpy vectorize function to achieve that, with a lambda function to store
# our temporary function.
fmap = np.vectorize(lambda t: size_dict[t])

# We then use fmap to the respective column of the numpy array (0) and to all the
# row present into the array itself. [:, 0]
X[:, 0] = fmap(X[:,0])

#printing first 5 row and col
X[:5]

# ONE HOT ENCODING
# We can't apply an order relation to the nominal categorical labels. We need to
# create additional dummy variables to represent the memebership of a color to
# his class (red, white, green etc).

# With pandas dataframe we use the method get_dummies from pandas: this method
# will scan for the appropriate column (in this case color) the number of unique
# elements present: it will then create N columns based on this number, replacing
# the original column and converting them to a boolean. We just need to pass the
# dataframe and the column name to the function.
shirts = pd.get_dummies(shirts, columns=["colore"])
shirts.head()

# For the numpy array the situation is different: we need scikit learn framework.
# There are two specific modules called LabelEncoder, useful for the encoding of
# the Categorical Ordinal features and the OneHotEncoder, useful for one hot
# encoding on a numpy array.
# From sklearn 0.22+, we can achieve that by a more simple and rapid way: Using
# ColumnTransformer: instead of encoding the label with LabelEncoder, we use
# directly one hot encoder and column transform to encode the N-1 classes to dummy
# boolean variables.
# Now, LabelEncoder is useless.
# from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Assigning the specific modules to some variables: label enc is now useless.
# label_enc = LabelEncoder()

# Creating the ColumnTransformer: we pass the name of the encoder we want to create,
# the constructor of the one hot encoder and the column we want to encode: in our
# case, color is the [1] column. We then pass remainder = 'passthrough' to say to the
# function that he must create ONLY the dummy leaving the rest intact.
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1])],remainder='passthrough')

# Here, LabelEncoder encode the target labels (i.e: Colors) to values between 0
# and N-1, where N are the number of N classes of object present inside the label.
# I.E: Before we had: Red, White, Green. We have then N-1 classes -> 3-1 = 2 classes.
# we'll then have values between 0-2 to represent red=0, green=1, white=2.

#  We're using the function fit_transform to achieve two things: Fit, and transforming
# the dataset. The Fitting operation is about "calculating the standard of the data",
# given by X' = (X - Mean) / (STD). Once fit has been applied, we then apply transform,
# that will remove all the NaN/None values into the dataset replacing them with X'.
# Note: Using ColumnTransformer those two lines are useless: we apply one hot encode
# directly on the array.
# X[:, 1] = label_enc.fit_transform(X[:,1])
# X[:20]

# We're still not done, because now we have our label encoding, having 0,1,2 as
# representer of the colors red, white green. But we still DO NOT KNOW which of them
# represent what. So we need to apply the One Hot Encoding method, creating 3 dummy
# fields to describe which color is what. We can achieve that with the created function
# ct obtained by ColumnTransformer. Since it returns a CRS MATRIX, we apply np.array
# function to it, that will return a numpy array: we then pass the function that returns
# the sparse matrix (ct.fit_transform) and specify which type we would like: float.
X = np.array(ct.fit_transform(X), dtype=np.float)

X[:20]
