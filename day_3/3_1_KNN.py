import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


# 1. load datasets
data = load_iris()
X = data.data
y = data.target

# 2. divide test && train dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 3. feature standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. transformer to pytorch tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# 5. difine class of KNN
class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        num_sample = X.shape[0]
        predictions = []

        for i in range(num_sample):
            distances = torch.norm(X[i] - self.X_train, dim=1)



# 6. create KNN instance and train
# 7. predict test dataset
# 8. calculate accuracy rate
# 9. visualization


