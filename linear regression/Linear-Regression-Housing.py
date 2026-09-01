# Part 1: Import Libraries

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Part 2: Load and Explore the Dataset

# Load the dataset
df = pd.read_csv("Housing.csv")

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())
# Part 3: Data Preprocessing & Feature Selection

# Select input features
X = df[["area", "bedrooms", "bathrooms", "stories", "parking"]]

# Select target variable
y = df["price"]

# Display the selected features
print("Input Features:")
print(X.head())

# Display the target variable
print("\nTarget Variable:")
print(y.head())
# Part 4: Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)
# Part 5: Create and Train Linear Regression Model

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("Linear Regression model trained successfully!")
# Part 6: Make Predictions

# Predict house prices using the test data
y_pred = model.predict(X_test)

# Display the first 5 predicted prices
print("Predicted House Prices:")
print(y_pred[:5])
# Part 7: Model Evaluation

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print("Model Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)
# Part 8: Coefficients and Interpretation

# Display coefficients for each feature
print("Feature Coefficients:")

for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", coefficient)

# Display intercept
print("\nIntercept:", model.intercept_)