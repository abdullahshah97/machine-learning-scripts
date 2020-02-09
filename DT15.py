import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, confusion_matrix
from sklearn import tree



df = pd.read_csv('2014train.csv') # Training data
dftest= pd.read_csv('2015test.csv') # Test Data


X = np.array(df.drop(['FTR','BbAvH','BbAvD','BbAvA'],1)) #features
y = np.array(df['FTR'])	#labels classes


Xtest = np.array(dftest.drop(['FTR','BbAvH','BbAvD','BbAvA'],1)) #features
ytest = np.array(dftest['FTR'])	#labels classes


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)


print(clf.score(Xtest,ytest))

scores = cross_val_score(clf, X, y, cv=10) # CV'tion
aveCV= (sum(scores)/len(scores))
print(aveCV)

results = clf.predict(Xtest)

#print(mean_absolute_error(ytest, results))

print(confusion_matrix(ytest, results))

#Exporting as DT
import graphviz

features= ["HomeTeam","AwayTeam","FTHG","FTAG"]
classes= ["-1","0","1"]

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=features, class_names= classes,filled=True, rounded=True) 
graph = graphviz.Source(dot_data) 

graph.render("Football14")