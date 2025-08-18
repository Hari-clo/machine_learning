import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv("ex4.csv")
df=pd.DataFrame(data)
x=df[['study_hr','attendence']]
y=df['result']
clf=DecisionTreeClassifier(criterion='entropy',random_state=0)
clf.fit(x,y)
plt.figure(figsize=(8,8))
plot_tree(clf,filled=True, feature_names=['study_hr','attendance'], class_names=["1","0"], fontsize=11)
plt.show()
new=[[5,85]]
pred=clf.predict(new)
print("Prediction for pass or fail","1" if pred[0]==1 else "0")
