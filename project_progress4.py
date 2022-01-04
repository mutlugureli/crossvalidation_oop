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

class BreastCancer():
    def __init__(self, pathway, column_to_predict):
        self.pathway = pathway
        self.column_to_predict = column_to_predict
        self.laoddataset()
        self.dropnulls()
        self.labelencoding()
        self.traindatapreprocess()
        self.scaling()

    def laoddataset(self):
        self.bc = pd.read_csv(self.pathway)

    def dropnulls(self):
        self.bc = self.bc.dropna(axis=1)
        self.bc = self.bc.drop(["id"], axis = 1)

    def labelencoding(self):
        label_quality = LabelEncoder()
        self.bc[self.column_to_predict] = label_quality.fit_transform(self.bc[self.column_to_predict])

    def traindatapreprocess(self):
        X = self.bc.drop(self.column_to_predict, axis=1) #quality sütununu ayırdık ve geri kalanlar x tablosu oldu.
        y = self.bc[self.column_to_predict]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

    def scaling(self):
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train) # nested array dönüyor. fit_transform does the math to fit all the values
                                    # so that all the variables affect the model at the same level.
        self.X_test = sc.transform(self.X_test) # we just transformed because we didn't want to calculate

class RandomForestClassifying(BreastCancer):
    def __init__(self, pathway, column_to_predict, n_estimators = 350):
        super().__init__(pathway, column_to_predict)
        self.n_estimators = n_estimators
        self.calculationRandomForestClassifying()
        self.reportsRFC()
        self.crossValidationRandomForest()

    def calculationRandomForestClassifying(self):
        self.rfc = RandomForestClassifier(self.n_estimators) # number of trees. Each tree will bring a prediction.
        self.rfc.fit(self.X_train, self.y_train)
        self.pred_rfc = self.rfc.predict(self.X_test)

class SupportVectorMachineClassifying(BreastCancer):
    def __init__(self, pathway, column_to_predict):
        super().__init__(pathway, column_to_predict)
        self.calculationSupportVectorMachineClassifying()
        self.reportsSVM()
        self.crossValidationSVC()

    def calculationSupportVectorMachineClassifying(self):
        self.clf = svm.SVC()
        self.clf.fit(self.X_train, self.y_train)
        self.pred_clf = self.clf.predict(self.X_test)

class MultilayerPerceptronClassifying(BreastCancer):
    def __init__(self, pathway, column_to_predict, hidden_layer_sizes=(2,2,2), max_iter=200):
        super().__init__(pathway, column_to_predict)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.calculationMultilayerPerceptronClassifying()
        self.reportsMLPC()
        self.crossValidationMLPC()

    def calculationMultilayerPerceptronClassifying(self):
        self.mlpc = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter)
#                     #3 hidden layers and size of each are 11
#                     #maximum iterations is 500. It means, it will go through the data 500 times.
        self.mlpc.fit(self.X_train, self.y_train)
        self.pred_mlpc = self.mlpc.predict(self.X_test)
