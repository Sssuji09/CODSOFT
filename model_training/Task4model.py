import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv('advertising.csv')

print("Dataset preview:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values in the dataset:")
print(df.isnull().sum())

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2)

X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_regressor = LinearRegression()

poly_regressor.fit(X_poly_train, y_train)

poly_y_pred = poly_regressor.predict(X_poly_test)

poly_mse = mean_squared_error(y_test, poly_y_pred)
poly_r2 = r2_score(y_test, poly_y_pred)

print("\nPolynomial Regression (Degree 2):")
print("Mean Squared Error:", poly_mse)
print("R²:", poly_r2)

joblib.dump(poly_regressor, 'polynomial_regression_model.pkl')
joblib.dump(poly, 'polynomial_features_model.pkl')

loaded_poly_regressor = joblib.load('polynomial_regression_model.pkl')
loaded_poly = joblib.load('polynomial_features_model.pkl')

example_input = np.array([[230.1, 37.8, 69.2]]) 
example_input_poly = loaded_poly.transform(example_input)
predicted_sales = loaded_poly_regressor.predict(example_input_poly)

print("\nPredicted Sales for example input using the saved model:", predicted_sales)

plt.scatter(y_test, poly_y_pred, color='blue')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales (Polynomial Regression)')
plt.show()
