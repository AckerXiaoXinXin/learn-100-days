import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


np.random.seed(6)


X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)


plt.scatter(X, y, color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()


model.fit(X_train, y_train)


print(model.intercept_)
print(model.coef_)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)

print(mse)


plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear')
plt.legend()
plt.show()