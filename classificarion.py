import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
#test
class Knn:
    def __init__(self,k,data):
        self.k = k
        self.data = data
    def distance(self,data,point):
        list_distances=list()
        for i in range(len(data)):
            distance= euclidean(data.iloc[i,:-1],point.iloc[:-1])
            list_distances.append([distance,data.iloc[i,-1]])
        list_distances = sorted(list_distances, key=lambda x: x[0])
        return list_distances
    def corss_validation(self,cv):
        
        list_cv= self.split_data(cv)
        
        accuracy=0
        
        for i in range(cv):
            list_cv2=list_cv.copy()
            test=list_cv2[i]
            list_cv2.pop(i)
            train=pd.concat(list_cv2,ignore_index=True)
            score=0
            for j in range(len(test)):
                list_distances= self.distance(train,test.iloc[j])  
                selected_class=self.classification(list_distances)
                if int(test.at[j,'Class'])==selected_class:
                    score+=1
            accuracy+=score/len(test)
            pass
        pass
        return accuracy/10
    def classification(self,list_distances):
        dict_count = dict()
        for i in range(self.k):
            dict_count[list_distances[i][1]]=dict_count.get(list_distances[i][1],0)+1
        max_value = max(dict_count, key=dict_count.get)
        return max_value
    def split_data(self,parts):
        list_cv=list()
        length = len(self.data)
        sub_length= length//parts
        rest= length % parts
        start=0
        index=np.random.choice(length, length, replace=False)
        for i in range(parts):
            data= self.data
            end = start + sub_length+(1 if i < rest else 0)
            test_data = index[start:end]
            test_data = data.iloc[test_data, :].reset_index(drop=True)
            
            list_cv.append(test_data)
        return list_cv
class Data:
    def __init__(self,data):
        self.data = data
    pass
    def split_data(self,percentage):
        length = len(self.data)
        train_number = int(percentage*length)
        index=np.random.choice(length, length, replace=False)
        train_data = index[:train_number]
        test_data = index[train_number:]
        train_data = self.data.iloc[train_data, :].reset_index(drop=True)
        test_data = self.data.iloc[test_data, :].reset_index(drop=True)
        return train_data, test_data