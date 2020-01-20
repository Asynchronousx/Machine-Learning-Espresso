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

# Before processing our data, we need to drop the columns that are useless: let's then drop the name column
titanic = titanic.drop("Name", axis=1)

# and, since we got the SEX column labeled as "male, female" we need to apply one hot encoding to the dataframe: in pandas,
# this is easily achieved by get_dummies function of pandas. We pass the titanic dataframe as argument.
titanic = pd.get_dummies(titanic)

# Let's build our numpy dataset array
X = titanic.drop("Survived", axis=1).values
Y = titanic["Survived"].values

# Let's split the dataset in train and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=520)

# Since we're going to use a DecisionTree, we got a big advantage: Decision Trees DO NOT need any kind of DATA SCALING,
# so we can avoid to use a standardization or a normalization.
# But, in short, what a decision tree is? A decision tree is a special data structure that holds in each node a question;
# in the root node we got the starting question, and from that, each tree link (connector between a node and another) will
# contain the answer of that specific question. The purpose of the decision tree is to reach the leafs (that will contain
# the right answer and prediction) as fast as possible. I.E: LOANS decision tree
#                           Income?
#            <50k /                       \ >50k
#          Seniority?                   Criminal records?
#    >10Y /         \ <10Y           Y /                 \ N
#  Allowed         Credit Card?     Not Allowed          Alloweed
#               Y /           \ N
#            Allowed      Not Allowed
#
# As we can see, the decision tree will, for each question, choose an answer: at each answer he will classify better the
# example he's analyzing, and, once reach a leaf, he will do the prediction.
# Note: Tree Deepness (major number of links connecting root to leaf) represent the model complexity.
# More deepness -> More complexity -> More overfitting.
# For further information on HOW decisions tree split and train their data, refer to some online source.
# Let's build our actual tree: We'll specify the IMPURITY CRITERION (Impurity are the metrics for check the goodness of
# information earnings, which will determine how good the prediction results will be.
# We got two metrics: Gini and Entropy: they're virtually the same thing, but Gini results in a more
# faster execution). Let's then use Gini as Criterion.
# We can also specify Tree max depth: As we stated, a deeper tree leads often to overfitting. To remove overfitting,
# lower the depth by using max_depth=N. Let's use 6 as max depth (could be lower or higher depending on the case).
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion="gini", max_depth=6)

# Train the model
tree.fit(X_train, Y_train)

# Predict both for test and train; we'll use the result to calculate the overfitting
Y_pred = tree.predict(X_test)
Y_pred_train = tree.predict(X_train)

# Check the overfitting by comparing those values with accuracy score: as we can see
# Without max_depth specified: TEST ACC - 0.77 / TRAIN ACC - 0.98
# With max_depth specified: TEST ACC - 0.80 / TRAIN ACC - 0.86 -> lower accuracy but
# lesser overfitting (because the model is less complex).
print("TREE: TEST ACCURACY {} - TRAIN ACCURACY {}".format(
      accuracy_score(Y_test, Y_pred),
      accuracy_score(Y_train, Y_pred_train)
))

# To check the results of our tree, let's utilize a sklearn module called graphviz: this module will build us
# a data model (contained into a dotfile) that will, with the right tool (webapp http://www.webgraphviz.com/ or
# here using graphviz methods), ready to be displayed and utilized.
# Let's build our DecisionTree using graphviz
from sklearn.tree import export_graphviz

# Now, let's open a file with the write right
treedot = open("treefile.dot", "w")

# using export_graphviz function, we pass in argument the tree we want to display, the output file created before,
# and the feature_names: we just use the columns of the titanic dataframe except from the survived (target).
export_graphviz(tree, out_file=treedot, feature_names=titanic.columns.drop("Survived"))

# remember to close the file
treedot.close()

# For visualizing it, we can use the webgraphviz webapp site, or use the method Source and View combined
# to visualize the tree:
from graphviz import Source
path = 'path/to/dot_file'
s = Source.from_file(path)
s.view()
