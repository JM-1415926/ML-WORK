------ Notes ------

* The dataset overview, including feature boxplots and missing data checks, is only available in Linear Regression.py. 

* The Decision Tree.py contains a commented-out section for generating a visualization of the tree. The generated tree is very complex, so it has been commented out. If you need modifications, please contact me for further optimization.

------ Requirements ------

*Python 3.10.8

*Libraries:

pandas
scikit-learn
matplotlib
numpy

------ Parameters and Pipeline ------

Decision Tree Regressor Model: DecisionTreeRegressor() from scikit-learn.
Parameters: Default parameters are used.

Linear Regression Model: LinearRegression() from scikit-learn.
Parameters: No regularization is applied (standard Linear Regression).

MLP Neural Network Model: MLPRegressor() from scikit-learn.
Parameters: A feed-forward neural network with one hidden layer (specific details are in the script)

Support Vector Machine (SVM)
Model: SVR(kernel='linear') from scikit-learn.
Parameters: Linear kernel is used to capture feature relationships.

------ Workflow ------

1. Load Data: The real estate dataset is loaded using Pandas.

3. Split Data: The dataset is split into features (X) and target (y).

3. Cross-Validation: The K-Fold cross-validation is applied to evaluate model performance on different subsets of data.

4. Training: Each model is trained on the training folds.

5. Prediction and Evaluation: Predictions are made for the test folds, and metrics (MSE and R² score) are calculated.

6. Visualization: Plots are generated to visualize the prediction accuracy and performance metrics for each fold.

------ Results ------

Mean Squared Error (MSE) and R² Score are calculated for each fold of cross-validation.
Scatter plots are generated to visualize the relationship between actual and predicted house prices.
Bar charts show MSE and R² scores across all folds




