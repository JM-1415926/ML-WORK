import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# Loading the dataset
data = pd.read_excel('Real estate valuation data set.xlsx')
X = data.drop(columns=['Y house price of unit area'])
y = data['Y house price of unit area']

# Setting up K-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

model = MLPRegressor(hidden_layer_sizes=(100, 70), max_iter=1000, random_state=42)

# storing the mean square error (MSE) and accuracy (R^2) per fold
mse_scores = []
accuracy_scores = []

# storing confusion martix
true_labels=[]
pred_labels=[]

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

    # for confusion martix
    for actual, predicted in zip(y_test, y_pred):
        if abs(actual - predicted) <= 5:
            true_labels.append('Accurate')
            pred_labels.append('accurate')
        elif predicted > actual:
            true_labels.append('Underestimate')
            pred_labels.append('Overestimate')
        else:
            true_labels.append('Overestimate')
            pred_labels.append('Underestimate')

    y_test = y_test.reset_index(drop=True)
    plt.scatter(y_test.values, y_pred, label=f'Fold{fold + 1}', alpha=0.6)

# output scatter plot
plt.plot([y.min(), y.max()], [y.min(), y.max()], label='Perfect Prediction')
plt.xlabel('Actual House Price per Unit Area')
plt.ylabel('Predicted House Price per Unit Area')
plt.legend()
plt.grid(True)
plt.show()


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

print(f'Mean Square Error per Fold Cross Validation: {mse_scores}')
print(f'Mean Mean Square Error: {np.mean(mse_scores)}')

print(f'Accuracy of cross validation per fold  (R^2): {accuracy_scores}')
print(f'Average accuracy (R^2): {np.mean(accuracy_scores)}')

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels, labels=['Accurate', 'Underestimate', 'Overestimate'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Accurate', 'Underestimate', 'Overestimate'])
disp.plot()
plt.title('confusion matrix')
plt.show()
