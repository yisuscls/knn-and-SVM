from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
column_names = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Class']
data = pd.read_csv(url, names=column_names, index_col='Id')

# Split the data into features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


Kernels = ['linear', 'poly', 'rbf', 'sigmoid']

best_accuracy=0
for k in Kernels:  
    svm = SVC(kernel=k)
    #cross validation
    scores = cross_validate(svm, X_train_scaled, y_train, cv=10)
    # accuracy of the cross validation
    accuracy = scores['test_score'].mean()
    #selection of the Kernel with the best score
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_kernel = k

# Evaluate the model on the test set
best_svm = SVC(kernel=best_kernel)
best_svm.fit(X_train_scaled, y_train)

# Evaluate the model on the test set
y_pred = best_svm.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"Best Kernel: {best_kernel}")
print(f"Training Accuracy with Best Kernel: {best_accuracy:.2f}")
print(f"Test Accuracy with Best Kernel: {test_accuracy:.2f}")
