import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the California housing dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
target = pd.Series(data.target, name='target')
df = pd.concat([df, target], axis=1)

# Exploratory Data Analysis
print(df.head())  # Display the first few rows of the dataset
print(df.describe())  # Summary statistics

# Data Visualization using Seaborn and Matplotlib
sns.pairplot(df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'target']])
plt.show()

# Data Preprocessing
X = df.drop('target', axis=1)  # Features
y = df['target']  # Target variable

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting with the model
predictions = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Visualizing predictions
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predicted Values')
plt.show()
