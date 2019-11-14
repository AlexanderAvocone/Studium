import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm


df = pd.read_csv("train.csv")
df2 = pd.read_csv("test.csv")
df3 = pd.read_csv("gender_submission.csv")


#auf NaN überprüfen
print(df.isnull().sum(),"\n")
print(df2.isnull().sum(),"\n")

#Test 1:
#Auffüllen von NaN
print("Mean of Age:",round(df["Age"].median(),2))
print("Standardabweichung:", np.std(df["Age"]),"\n")
""" Aus den Rohdaten sehen wir, dass die NaN ein Mister im Namen haben
--> Erwachsene """
df["Age"] = df["Age"].fillna(df["Age"].median())
df2["Age"] = df2["Age"].fillna(df2["Age"].median())
df["Embarked"].fillna(value = "S", inplace =True)
df2["Embarked"].fillna(value = "S", inplace = True)


#Dictionary für Sex
sex = {"male":0,"female":1}
df["Sex"] = df["Sex"].apply(lambda  x:sex[x])
df2["Sex"] = df2["Sex"].apply(lambda  x:sex[x])


#Dictionary für Embarked
embarked = {"S":0,"Q":1,"C":2}
df["Embarked"] = df["Embarked"].replace(embarked)
df2["Embarked"] = df2["Embarked"].replace(embarked)





train1 = df[["Pclass"]]
train2 = df2[["Pclass"]]
clf = svm.LinearSVC()


#Training
x = clf.fit(train1,df["Survived"])

#Predicting
end = clf.predict(train2)


print(end)



"""
#Saving prediction as CSV without index
df3["Survived"] = end
df3.set_index("PassengerId", inplace = True)
df3.to_csv("result.csv")

"""













