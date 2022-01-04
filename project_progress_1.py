from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm



wine = pd.read_csv(r"C:\\Users\\Monster PC\Downloads\winequality-red.csv")


#
#
# for i in wine.columns:
#     print(i)



# # print(wine)
#
# print(wine.head())
# print(wine.info())

# #
# print(wine.isnull().sum())
#
#
# preprocessing data
# we need a value to determine the quanlity of wine is good or not.
# here I chose "6" for it. It means, if the quality is less than 6, than it is "bad" but if it is more than 6,
# than it is a good quality wine.
#
bins = (2,5,8)
group_names = ["bad","good"]
#
wine["quality"] = pd.cut(wine["quality"], bins=bins, labels=group_names)
#
print(wine)


le = LabelEncoder()
wine["quality"] = le.fit_transform(wine["quality"])

print(wine)


# now we seperate the dataset as response variable and feature variable
X = wine.drop("quality", axis =1) #quality sütununu ayırdık ve geri kalanlar x tablou oldu.
y = wine["quality"]

print(X)
print(y)


#Train and test splitting of data



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train)
print("\n")


# Setting random_state a fixed value will guarantee that same sequence of random numbers are generated
# each time we run the code. And unless there is some other randomness present in the process, the results
# produced will be same as always. This helps in verifying the output.
#
#
#
#
#
sc= StandardScaler() # nested array dönüyor.
                     # z = (x - u) / s
                     # z = new value after calculation
                     # x = old value
                     # u = mean
                     # s = standard deviation


X_train = sc.fit_transform(X_train) # nested array dönüyor. fit_transform does the math to fit all the values
                                    # so that all the variables affect the model at the same level.
print (X_train)
#
X_test = sc.transform(X_test) # we just transformed because we didn't want to calculate


print(X_test)
                              # mean and standard deviation again.
#
#
#
# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=200) # number of trees. Each tree will bring a prediction.
rfc.fit(X_train, y_train)

pred_rfc = rfc.predict(X_test)


print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))
#
#
# # #
# #
# #
# # Support Vector Machine Classifier
clf = svm.SVC()

clf.fit(X_train, y_train)
pred_clf = clf.predict(X_test)
print(classification_report(y_test, pred_clf))
print(confusion_matrix(y_test, pred_clf))


# #
# Neural Network (it is generally needed when you are working w huge amount or continueous flowing of data)
# Multilayer Perceptron Classifier


mlpc = MLPClassifier(hidden_layer_sizes=(11,11,11), max_iter=500)
                    #3 hidden layers and size of each are 11
                    #maximum iterations is 500. It means, it will go through the data 500 times.
# # #
mlpc.fit(X_train,y_train)
pred_mlpc = mlpc.predict(X_test)
# # #
# # # total = 0
# # # validation = cross_val_score(clf, X_train, y_train)
# # # for i in validation:
# # #     total = total + i
# # #
# # # print (total/len(validation))
# # #
# # #
# # #
print (classification_report(y_test, pred_mlpc))
print(confusion_matrix(y_test, pred_mlpc))
# # # #
# # # # cm = accuracy_score(y_test, pred_rfc)
# # # # Xnew = [[11.6,0.53,0.66,3.65,0.121,6.0,14.0,0.9978,3.05,0.74,11.5]]
# # # # Xnew = sc.transform(Xnew)
# # # # ynew = rfc.predict(Xnew)
# # # # print(ynew)
# # # #
# # # #
# # # # cm = accuracy_score(y_test, pred_mlpc)
# # # # Xnew = [[11.6,0.53,0.66,3.65,0.121,6.0,14.0,0.9978,3.05,0.74,11.5]]
# # # # Xnew = sc.transform(Xnew)
# # # # ynew = rfc.predict(Xnew)
# # # # print(ynew)
# # # #
# # # #
Xnew = [[9.6,0.53,0.66,3.65,0.121,6.0,14.0,0.9978,3.05,0.74,10.0]]
Xnew = sc.transform(Xnew)
ynew = rfc.predict(Xnew)
print(ynew)
