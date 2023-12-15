import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
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
print(X_train,X_test,end="\n")
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Implement K-nearest neighbor classifier with 10-fold cross-validation
best_accuracy = 0
best_k = 0
ks=list()
for k in range(1, 25):  
    knn = KNeighborsClassifier(n_neighbors=k)
    #cross validation
    scores = cross_validate(knn, X_train_scaled, y_train, cv=10)
    # accuracy of the cross validation
    accuracy = scores['test_score'].mean()
    ks.append([k, accuracy])
    #selection of the K with the best score
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k
        

# Train the model with the best k value
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)

# Evaluate the model on the test set
y_pred = best_knn.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)



print(f"Best K: {best_k}")
print(f"Training Accuracy with Best K: {best_accuracy:.2f}")
print(f"Test Accuracy with Best K: {test_accuracy:.2f}")
for i in ks:
    print(f" K: {i[0]}, Score: {i[1]}")
    

