# linear-regression
we import the necessary libraries required for the Linear Regression project. Pandas is used to load and handle the dataset, Matplotlib is used to create graphs and visualizations, and Scikit-learn is used to split the data, build the Linear Regression model, and evaluate its performance.

we load the Housing Price Prediction dataset using Pandas and explore its basic structure. We display the first few rows, dataset information, and check for missing values. This helps us understand the data before building the Linear Regression mode 

we select the input features and target variable for the Linear Regression model. The selected features are area, bedrooms, bathrooms, stories, and parking. The target variable is price, which the model will predict

 we divide the dataset into training and testing sets. The training data is used to train the Linear Regression model, while the testing data is used to evaluate how well the model performs on unseen data. We use 80% of the data for training and 20% for testing.

we create a Linear Regression model using Scikit-learn. The model is trained using the training data so that it can learn the relationship between the selected housing features and the house price.

we use the trained Linear Regression model to predict house prices for the testing data. The model uses the selected features from X_test and generates predicted prices, which will later be compared with the actual prices to evaluate the model.

we evaluate the performance of the Linear Regression model using three evaluation metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score. MAE and MSE measure the prediction error, while the R² Score shows how well the model explains the variation in house prices.

we display the coefficients of the Linear Regression model. Each coefficient represents how a particular feature affects the predicted house price while keeping the other features constant. We also display the intercept of the model.

 we visualize the actual and predicted house prices using a scatter plot. This helps us compare the model's predictions with the actual values and understand how well the Linear Regression model performs.
 
