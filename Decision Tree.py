from sklearn.model_selection import cross_val_score, train_test_split # Added train_test_split import
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Load Excel file
data = pd.read_excel('Real estate valuation data set.xlsx')

# Define features and target
X = data.drop(columns=['Y house price of unit area'])
y = data['Y house price of unit area']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Setting up K-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

model = DecisionTreeRegressor()

# storing the mean square error (MSE) and accuracy (R^2) per fold
mse_scores = []
accuracy_scores = []

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    # Segmentation of training and test sets based on indexing
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Calculate the mean square error (MSE) and store
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

    # Calculate the accuracy (R^2) and store
    accuracy = r2_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

    y_test = y_test.reset_index(drop=True)
    plt.scatter(y_test.values, y_pred, label=f'Fold{fold+1}', alpha=0.6)

# output scatter plot
plt.plot([y.min(), y.max()], [y.min(), y.max()], label='Perfect Prediction')
plt.xlabel('Actual House Price per Unit Area')
plt.ylabel('Predicted House Price per Unit Area')
plt.legend()
plt.grid(True)
plt.show()

# plt.figure(figsize=(20, 10))
# plot_tree(model, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
# plt.show()

# Plotting for each fold
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(mse_scores) + 1), mse_scores, color='skyblue')
plt.xlabel('Fold')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Mean Squared Error per Fold')
plt.xticks(range(1, len(mse_scores) + 1))
plt.show()
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(accuracy_scores) + 1), accuracy_scores, color='lightgreen')
plt.xlabel('Fold')
plt.ylabel('R^2 Score')
plt.title('R^2 Score per Fold')
plt.xticks(range(1, len(accuracy_scores) + 1))
plt.show()

# Perform cross-validation
dt_cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print('Decision Tree Cross-validated MSE:', -dt_cv_scores.mean())

# Make predictions on the test set
y_dt_pred = model.predict(X_test)
dt_mse = mean_squared_error(y_test, y_dt_pred)
print('Decision Tree MSE on test set:', dt_mse)


