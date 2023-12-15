import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from classificarion import Data,Knn
# Load the dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
column_names = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Class']
data = pd.read_csv(url, names=column_names, index_col='Id')
data=Data(data)
training,testing = data.split_data(0.3)
errors=[]

for k in range(1,21):
    knn=Knn(k,training)
    score=knn.corss_validation(10)
    errors.append([k,score])
kmax=0
scoreMax=0
for [best_k,score] in errors:
    count=0
    if(score>0.9):
        knn=Knn(best_k,training)
        for  i in range( len(testing)):
            distances=knn.distance(training,testing.iloc[i])
            classification=knn.classification(distances)
            if(classification==testing.at[i,'Class']):
                count+=1
        count=count/len(testing)
        if count>scoreMax:
            scoreMax=count
            kmax=best_k
        print(kmax,scoreMax)            