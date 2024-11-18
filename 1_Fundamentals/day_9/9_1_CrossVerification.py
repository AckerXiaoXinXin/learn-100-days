import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier


iris = load_iris()
X, y = iris.data, iris.target

kf = KFold(n_splits=5, shuffle=True, random_state=42)

classifier = DecisionTreeClassifier()

scores = cross_val_score(classifier, X, y, cv=kf)

print(f'score: {scores}')

print(f'mean score: {np.mean(scores)}')


