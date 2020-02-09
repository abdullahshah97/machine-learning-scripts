import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,confusion_matrix

df = pd.read_csv('2014train.csv') # Training data

dftest= pd.read_csv('2015test.csv') # Test Data


X = np.array(df.drop(['FTR','BbAvH','BbAvD','BbAvA'],1)) #features
y = np.array(df['FTR'])	#labels classes

Xtest = np.array(dftest.drop(['FTR','BbAvH','BbAvD','BbAvA'],1)) #features
ytest = np.array(dftest['FTR'])	#labels classes

reg = linear_model.LinearRegression()
reg.fit (X, y)

print(reg.score(Xtest,ytest))

scores = cross_val_score(reg, X, y, cv=10) # CV'tion
aveCV= (sum(scores)/len(scores))
print(aveCV)

results = reg.predict(Xtest)
results = [round(x) for x in results]
for n, i in enumerate(results):
    if i == -2:
        results[n] = -1
    if i == 2:
        results[n] = 1
    if i == 3:
        results[n] = 1


#print(mean_absolute_error(ytest, results))

print(confusion_matrix(ytest, results))

# File writing
#results = reg.predict(Xtest)
#results = [round(x) for x in results]
#for n, i in enumerate(results):
#    if i == -2:
#        results[n] = -1
#    if i == 2:
#        results[n] = 1
#    if i == 3:
#        results[n] = 1

#df1 = pd.read_csv('2014prac.csv')
#new_column = pd.DataFrame({'LR Results Rounded': results})
#df1 = df1.merge(new_column, how= 'right', left_index = True, right_index = True)
#df1.to_csv('2014prac.csv',index=False)

#plt.scatter(X_test,y,  color='black')
#plt.plot(X_test, y, color='blue', linewidth=3)

#plt.xticks(())
#plt.yticks(())

#plt.show()