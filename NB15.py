import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix

df = pd.read_csv('2014train.csv') # Training data

dftest= pd.read_csv('2015test.csv') # Test Data


X = np.array(df.drop(['FTR','BbAvH','BbAvD','BbAvA'],1)) #features
y = np.array(df['FTR'])	#labels classes

Xtest = np.array(dftest.drop(['FTR','BbAvH','BbAvD','BbAvA'],1)) #features
ytest = np.array(dftest['FTR'])	#labels classes


clf= GaussianNB()#New Classifier


clf.fit(X, y)#Fit on train data
print(clf.score(Xtest,ytest))

#Validations
scores = cross_val_score(clf, X, y, cv=10) # CV'tion
aveCV= (sum(scores)/len(scores))
print(aveCV)

results = clf.predict(Xtest)

#print(mean_absolute_error(ytest, results))

print(confusion_matrix(ytest, results))

# File writing
#results = clf.predict(Xtest)
#df1 = pd.read_csv('2014prac.csv')
#new_column = pd.DataFrame({'NB Results': results})
#df1 = df1.merge(new_column, how= 'right', left_index = True, right_index = True)
#df1.to_csv('2014prac.csv',index=False)