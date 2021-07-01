#import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
#reading data and assigning values to dependent and independent variables
dataset=pd.read_csv('ckd_dataset.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
print(dataset.iloc[0:20, 0:10])
 
#forming a dataframe
df = pd.DataFrame(dataset) 
 
#trim white spaces
df=df.applymap(lambda x: x.strip() if isinstance(x, str) else x) 
 
#converting object datatype to float datatype
df['pcv'] = pd.to_numeric(df.pcv, errors='coerce')
df['wc'] = pd.to_numeric(df.wc, errors='coerce')
df['rc'] = pd.to_numeric(df.rc, errors='coerce')
 
#dealing with null/missing values
df=df.bfill(axis ='rows')
 
one_hot = pd.get_dummies(df['rbc'],  prefix='rbc')
# Drop column B as it is now encoded
df = df.drop('rbc',axis = 1)
# Join the encoded df
df = df.join(one_hot)
 
one_hot = pd.get_dummies(df['pcc'],  prefix='pcc')
df = df.drop('pcc',axis = 1)

df = df.join(one_hot)
one_hot = pd.get_dummies(df['pc'],  prefix='pc')
df = df.drop('pc',axis = 1)
df = df.join(one_hot)
one_hot = pd.get_dummies(df['ba'],  prefix='ba')
df = df.drop('ba',axis = 1)
df = df.join(one_hot)
one_hot = pd.get_dummies(df['htn'],  prefix='htn')
df = df.drop('htn',axis = 1)
df = df.join(one_hot)
one_hot = pd.get_dummies(df['dm'],  prefix='dm')
df = df.drop('dm',axis = 1)
df = df.join(one_hot)
one_hot = pd.get_dummies(df['cad'],  prefix='cad')
df = df.drop('cad',axis = 1)
df = df.join(one_hot)
one_hot = pd.get_dummies(df['pe'],  prefix='pe')
df = df.drop('pe',axis = 1)
df = df.join(one_hot)
one_hot = pd.get_dummies(df['ane'],  prefix='ane')
df = df.drop('ane',axis = 1)
df = df.join(one_hot)
one_hot = pd.get_dummies(df['appet'],  prefix='appet')
df = df.drop('appet',axis = 1)
df = df.join(one_hot)
 
col1=df['classification']
df = df.drop('classification',axis = 1)
df = df.join(col1)
 
X=df.iloc[:,1:-1].values
Y=df.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder
Y=LabelEncoder().fit_transform(Y)
#splitting data into train and test data
from sklearn.model_selection import train_test_split as tts
X_train,X_test,Y_train,Y_test=tts(X,Y,test_size=0.3,random_state=0)
X_test,Y_test
 
#scaling data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train[ : ,0:14]=sc.fit_transform(X_train[:,0:14])
X_test[:,0:14]=sc.transform(X_test[:,0:14])
#data preprocessing code ends

# random subspace model building
scores1=[] ; scores2=[]; scores3=[];
from sklearn.tree import DecisionTreeClassifier
decisiontreeclassifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
dtree=BaggingClassifier(decisiontreeclassifier,bootstrap=False,bootstrap_features=True,max_features=20)
dtree.fit(X_train,Y_train)
scores1=dtree.predict(X_test)
print(scores1)
 
#performance evaluation code for scores1(dtree)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,scores1)
cm
 
from sklearn.neighbors import KNeighborsClassifier
knnclassifier=KNeighborsClassifier(n_neighbors=3,metric='minkowski',p=3)
knn=BaggingClassifier(knnclassifier,bootstrap=False,bootstrap_features=True,max_features=10)
knn.fit(X_train,Y_train)
scores2=(knn.predict(X_test))
print(scores2)
 
#performance evaluation code for scores2(knn)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,scores2)
cm
 
from sklearn.naive_bayes import GaussianNB
nbclassifier=GaussianNB()
nb=BaggingClassifier(nbclassifier,bootstrap=False,bootstrap_features=True,max_features=20)
nb.fit(X_train,Y_train)
scores3=(nb.predict(X_test))
print(scores3)
 
#performance evaluation code for scores3(naive bayes)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,scores3)
cm
 
from sklearn.ensemble import VotingClassifier
estimator = [] 
 
estimator.append(('DTC', dtree))
estimator.append(('KNN', knn)) 
estimator.append(('NB', nb))

# Voting Classifier with hard voting 
vot_hard = VotingClassifier(estimators = estimator, voting ='hard') 
vot_hard.fit(X_train, Y_train) 
y_pred = vot_hard.predict(X_test) 
 
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)
cm

#performance evaluation code starts
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)
cm
#accuracy
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
#kappa score
from sklearn.metrics import cohen_kappa_score, make_scorer
kappa = cohen_kappa_score(Y_test, Y_pred, labels=None)
print("Kappa value: ", kappa)
#sensitivity
sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )
#specificity
specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)
#ROC curve
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import auc, roc_auc_score, roc_curve
 
fpr, tpr, threshold = roc_curve(Y_test, Y_pred)
roc_auc = auc(fpr, tpr)
plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('roc.png')
plt.show()
#performance evaluation code ends