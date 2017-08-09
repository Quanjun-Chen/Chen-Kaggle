
# coding: utf-8

# In[123]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[124]:

data=pd.read_csv('./train.csv')
data.head(1)


# In[125]:

y=data['Survived']
pid=data['PassengerId']
pclass = pd.get_dummies( data.Pclass , prefix='Pclass' )
sex = pd.get_dummies( data.Sex , prefix='Sex' )


# In[126]:

ma = data['Age'].mean()


# In[133]:

temp=data['Age'].fillna(ma)
child=temp<14


# In[134]:

x=pd.concat([pclass,sex,child,data['Fare']],axis=1)


# In[135]:

child.head(1)


# In[136]:

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)


# In[137]:

clf = GradientBoostingClassifier()
clf.fit(train_x, train_y) 
y_pred = clf.predict(test_x)
print accuracy_score(test_y, y_pred)


# In[114]:

data=pd.read_csv('./test.csv')

pid=data['PassengerId']
pclass = pd.get_dummies( data.Pclass , prefix='Pclass' )
sex = pd.get_dummies( data.Sex , prefix='Sex' )
ma = data['Age'].mean()
temp=data['Age'].fillna(ma)
child=temp<14
mf = data['Fare'].mean()
data['Fare']=data['Fare'].fillna(mf)
x=pd.concat([pclass,sex,child,data['Fare']],axis=1)


# In[115]:

y_pred = clf.predict(x)


# In[116]:

output=pd.DataFrame( { 'PassengerId': pid , 'Survived': y_pred } )


# In[117]:

output.to_csv('./output.csv',index=False)


# In[ ]:



