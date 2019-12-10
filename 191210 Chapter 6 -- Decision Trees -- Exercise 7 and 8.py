# Common imports
import numpy as np


#7. Train and fine-tune a Decision Tree for the moons dataset by following
#these steps:


#a. Use make_moons(n_samples=10000, noise=0.4) to generate
#a moons dataset.

from sklearn.datasets import make_moons

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)


#b. Use train_test_split() to split the dataset into a training
#set and a test set.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#c. Use grid search with cross-validation (with the help of the
#GridSearchCV class) to find good hyperparameter values for a
#DecisionTreeClassifier. Hint: try various values for
#max_leaf_nodes.

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)

grid_search_cv.fit(X_train, y_train)

grid_search_cv.best_estimator_

#d. Train it on the full training set using these hyperparameters,
#and measure your model’s performance on the test set. You
#should get roughly 85% to 87% accuracy.
#

from sklearn.metrics import accuracy_score

y_pred = grid_search_cv.predict(X_test)
accuracy_score(y_test, y_pred)


#%% 8. Grow a forest by following these steps:

#a. Continuing the previous exercise, generate 1,000 subsets of the
#training set, each containing 100 instances selected randomly.
#Hint: you can use Scikit-Learn’s ShuffleSplit class for this.

from sklearn.model_selection import ShuffleSplit

n_trees = 1000
n_instances = 100

mini_sets = []

rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances, random_state=42)

for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))
    
    
#b. Train one Decision Tree on each subset, using the best
#hyperparameter values found in the previous exercise. Evaluate
#these 1,000 Decision Trees on the test set. Since they were
#trained on smaller sets, these Decision Trees will likely
#perform worse than the first Decision Tree, achieving only
#about 80% accuracy.

from sklearn.base import clone

forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]

accuracy_scores = []

for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)
    
    y_pred = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    
np.mean(accuracy_scores)

#c. Now comes the magic. For each test set instance, generate the
#predictions of the 1,000 Decision Trees, and keep only the
#most frequent prediction (you can use SciPy’s mode() function
#for this). This approach gives you majority-vote predictions
#over the test set.

Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)
Y_pred.shape

for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)
    
Y_pred[:10]

from scipy.stats import mode

y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)

#d. Evaluate these predictions on the test set: you should obtain a
#slightly higher accuracy than your first model (about 0.5 to
#1.5% higher). Congratulations, you have trained a Random
#Forest classifier!

accuracy_score(y_test, y_pred_majority_votes.reshape([-1]))
