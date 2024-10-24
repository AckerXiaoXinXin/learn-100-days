from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

classifier = RandomForestClassifier(n_estimators=100, random_state=32)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')

print(classification_report(y_test, y_pred, target_names=iris.target_names))

ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test, display_labels=iris.target_names)
plt.title('Confusion matrix')
plt.show()
