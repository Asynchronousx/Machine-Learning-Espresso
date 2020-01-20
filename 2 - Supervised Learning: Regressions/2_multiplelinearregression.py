import pandas as pd
import numpy as np

boston = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                     sep='\s+',
                     names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PRATIO","B","LSTAT","MEDV"])

boston.head()

# Print some info
boston.info()

# Useful to check if data will need to be standardized later on: we need to standardize/normalize
# based on if there are HUGE difference between values.
boston.describe()

# Having multiple properties, means that we should choose which of them will be useful for our
# linear regression model to predict better the final target. To do that, we must establish
# the correlation between the multiple properties in our dataset. To achieve that, we can use
# the function corr of the dataframe, to check the correlation bound of each of them.
# The way we read correlation is easy:
# Values trending to -1 mean an inverse correlation: increasing one, the other will decrease.
# Values trending to 0 mean a poor correlation: increasing or decreasing one, will poorly affect the other.
# Values trending to 1 mena a direct correlation: increasing one will increase the other, same for decreasing.

# The function corr will return a Correlation matrix. A correlation matrix is a table showing, in each cell,
# values called correlation coefficent that shows bound between sets of variables.
# Each random cell Xi is correlated with each of other values in the matrix at Xj.
# This allow us to see which pair of variable got the high correlation.
# NOTE: We do not confuse correlation and covariance:
# A measure used to indicate how two variables changes toghether is knows as covariance.
# A measure used to represent how strongly two variables are bounded, is known as correlation.
# Additional: Correlation matrix is triangular.

boston.corr()

# Sometimes can be Harsh visualizing correlation between variables: we can then use a tool called
# Heatmap to show them more clearly. To achieve that simply, we can use Seaborn, an elegant tool
# to build heatmap.
import seaborn as sns

# To build our heatmap we can use the function heatmap of seaborn: the input arguments are
# the correlation matrix, and the columns name, both for x and y axis:
# White color means high correlation, Black means inverse correlation, Red means poor correlation.
sns.heatmap(boston.corr(),
            xticklabels=boston.columns,
            yticklabels=boston.columns)

# Now let's take the columns that seems to have both an high correlation and an inverse
# correlation to see more in specific the bound between property
corr_col = ["RM", "LSTAT", "PRATIO", "TAX", "INDUS", "MEDV"]

# And let's rebuild our heatmap passing in input just those columns: we'll add also a
# numeric values to each of the cell and specifying a size of the text with the argument
# annot=true for displaying numeri values and annot_kws={"size":12} for specifying the
# text size. Assure to pass the right columns to each of boston dataframe to correct visualization.
sns.heatmap(boston[corr_col].corr(),
            xticklabels=boston[corr_col].columns,
            yticklabels=boston[corr_col].columns,
            annot=True,
            annot_kws={'size':12})

# Now, to have some more information about the pair correlation between those variables,
# we can use the function pairplot of seaborn, passing in input just the dataframe (reduced
# with the desired columns) to check the plotted correlation between each pair of property.
# The plot are shown between each pair of value Row-Column for each pair of our subset.
# Just like the heatmap above, but instead of a colored square with a number, we got an entire
# plot.
sns.pairplot(boston[corr_col])

# Since we saw that RM and LSTAT are the most correlated (direct for RM, inverse for LSTAT), we can
# now build our linear regression model on those properties. As usual, we use MEDV for the target
# (since we want to predict the $ value) and RM and LSTAT as property. Let's then build our numpy
# array following this schema.
X = boston[["RM", "LSTAT"]].values
Y = boston["MEDV"].values

# Check the shape to decide which will be the size of test/train set
X.shape
Y.shape


# Let's build our train and test set using the appropriate function.
# Since we're going we're going to validate our train over multiple process (next
# we'll do a linear regression on the entire dataset) we use a new flag called random_state
# that will act as a FIXED random number generated: in next train runs it will generate the exact
# random number sequences to keep consistency between different train sessions and assure that
# train should be equal to another, keeping the same result with different sets.
# The random state can be initialized to any integer.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=420)

# And now let's build our actual linear regression model.
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# fit the model to build internal hyperparamters: obviously we need to pass the
# train sets to train our model, both X and Y.
lr.fit(X_train, Y_train)

# And now let's predict the missing values using the property test set
Y_pred = lr.predict(X_test)

# As good practice, let's now check the score of the train (using r2 scoring) and the coefficent
# (both bias and weights) of the training done.

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Evalute goodness of our accuracy using those functions: we obviously need to pass
# in input the Y_test set and the Y_pred set, to check how much they're "distant"
# from each other in terms of accuracy.
r2_score(Y_test, Y_pred)
mean_squared_error(Y_test, Y_pred)

# Checking coefficent bias and weight
lr.coef_
lr.intercept_

# Done the training on just RM and LSTAT as property, we now want to train our model
# on the entire dataset. We then construct the respective numpy arrays.
# X (property) will contains all the property except for MEDV (the target) and Y
# (target) will contain just MEDV.
Xe = boston.drop("MEDV", axis=1)
Ye = boston["MEDV"]

# Before creating test and train dataset, we should ask ourself: Are there values MUCH
# bigger than others? This should be the case since we're operating on an entire datasets.
# For linear models we need to STANDARDIZE data to increase accuracy. We use the standardization
# methods offered by sklearn to achieve that.
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

# Think about that: on what we should use standardization and when? The answer is pretty simple:
# We should use standardadization only on the dataset we like to "scale" on the same similar values;
# this dataset is obviously the property dataset, since it contains various values with various sizes.
# Now, when data should be standardized? This is another pretty easy answer. We can standardize the entire
# property dataset, and then split it into train and test set. But, thinking about that, since with the splitting
# we are just dividing the dataset into two parts, the resultant train and test set will contains different
# values inside them, and that if we apply a standardization on them the result will be different. So we PREFER
# to standardize the test and train set instead, but we can also do that on the entire set. It's a choice.
# Usually STD and NORM are done on the property train/test set.
# Now, splitting the dataset:
Xe_train, Xe_test, Ye_train, Ye_test = train_test_split(Xe, Ye, test_size=0.3, random_state=420)

# Now, let's standardize the property test and train set
Xe_train_std = ss.fit_transform(Xe_train)

# Since we've already fitted the standardization model we could just use transform now to avoid time wastes
Xe_test_std = ss.transform(Xe_test)

# Let's now build our linear regression model
lre = LinearRegression()
lre.fit(Xe_train_std, Ye_train)
Ye_pred = lre.predict(Xe_test_std)

# Judge the goodness of the predicted dataset using r2 and ESS
r2_score(Ye_test, Ye_pred)
mean_squared_error(Ye_test, Ye_pred)

# Checking coefficent bias and weights
lre.coef_
lre.intercept_
