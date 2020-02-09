import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, confusion_matrix

df = pd.read_csv('2014train.csv') # Training data

dftest= pd.read_csv('2015test.csv') # Test Data


X = np.array(df.drop(['FTR','BbAvH','BbAvD','BbAvA'],1)) #features
y = np.array(df['FTR'])	#labels classes

Xtest = np.array(dftest.drop(['FTR','BbAvH','BbAvD','BbAvA'],1)) #features
ytest = np.array(dftest['FTR'])	#labels classes


clf= neighbors.KNeighborsClassifier(n_neighbors =19)#New Classifier


clf.fit(X, y)#Fit on train data

print(clf.score(Xtest,ytest))

scores = cross_val_score(clf, X, y, cv=10) # CV'tion
aveCV= (sum(scores)/len(scores))
print(aveCV)


results = clf.predict(Xtest)



#print(mean_absolute_error(ytest, results))
print(confusion_matrix(ytest, results))

# File writing
#results = clf.predict(Xtest)
#df1 = pd.read_csv('2014prac.csv')
#new_column = pd.DataFrame({'KNN Results': results})
#df1 = df1.merge(new_column, how= 'right', left_index = True, right_index = True)
#df1.to_csv('2014prac.csv',index=False)












#from sklearn.metrics import mean_squared_error

#y_true= [0,1,1,-1,1,1]
#y_pred = clf.predict([[0,0,0,0,15,8,3,3,12,13,10,7,2,4,0,0],[0,2,0,0,6,14,2,4,6,9,3,10,0,2,0,0],[0,1,0,0,8,29,2,9,9,6,1,6,2,3,0,0],[0,2,0,0,6,18,3,6,6,10,5,7,1,2,1,0],[4,0,1,0,22,9,6,1,19,7,11,1,2,2,0,0],[4,3,1,1,16,11,7,4,10,7,5,6,2,3,0,0]])

#print(mean_squared_error(y_true, y_pred))



#print(clf.predict([[0,0,0,0,15,8,3,3,12,13,10,7,2,4,0,0],[0,2,0,0,6,14,2,4,6,9,3,10,0,2,0,0],[0,1,0,0,8,29,2,9,9,6,1,6,2,3,0,0],[0,2,0,0,6,18,3,6,6,10,5,7,1,2,1,0],[4,0,1,0,22,9,6,1,19,7,11,1,2,2,0,0],[4,3,1,1,16,11,7,4,10,7,5,6,2,3,0,0]]))

#example= np.array([4,3,2,2,27,6,10,3,9,12,9,4,0,1,0,0])
#example=example.reshape(1,-1)
#prediction = clf.predict(example)
#print(prediction)
#example= np.array([0,0,0,0,29,4,2,0,10,13,13,0,2,1,0,0])
#example=example.reshape(1,-1)
#prediction = clf.predict(example)
#print(prediction)
#p=2 and minkowski makes it standard euclidean
#This algorithm produces a predictive accuracy between 0.5-0.7 degrees of accuracy
#Could be improved through changing values such as
#test_size (training and test), knn, weights uniform or distance, decide to drop some columns
# Possibly add some columns, take into account team names

#Distance appears to work better for knn 20, test size 0.3 avg acc == 0.6-0.65 

#Decide on k by displaying a graph for average cv error for 20 try20