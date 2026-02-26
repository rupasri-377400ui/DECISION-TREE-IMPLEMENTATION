import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree 
from sklearn.metrics import accuracy_score,classification_report
iris=load_iris()
x=iris.data
y=iris.target
print("Feature names:",iris.feature_names)
print("Target Names:",iris.target_names)
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
dt_model=DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=42)
dt_model.fit(X_train,y_train)
y_pred=dt_model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
print("\nClassification Report:")
print(classification_report(y_test,y_pred))
plt.figure(figsize=(20,10))
plot_tree(dt_model,feature_names=iris.feature_names,class_names=iris.target_names,filled=True,rounded=True,fontsize=10)
plt.show()
