import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


np.random.seed(0)

iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
#print(df.head(10))
#print(iris)
df['spicies']=pd.Categorical.from_codes(iris.target,iris.target_names)
#print(df.head())

df['is_train0']=np.random.uniform(0,1,len(df))<=.75
print(df.head())

train,test=df[df['is_train0']==True],df[df['is_train0']==False]
print("no of obs in train",len(train))
print("no of obs in test",len(test))


feature=df.columns[:4]
print(feature )

y= pd.factorize(train['spicies'])[0]
print(y)

clf=RandomForestClassifier(n_jobs=2,random_state=0)
clf=clf.fit(train[feature],y)
pred=clf.predict(test[feature])
pt=clf.predict_proba(test[feature])
print(pt)
preds=iris.target_names[pred]
print(preds[20:30])
print(pd.DataFrame({"actual":test['spicies'],"predicted":preds}))
tab=pd.crosstab(test['spicies'],preds,rownames=['actual'],colnames=['predicted'])
print(tab)