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

class WineQuality():
    def __init__(self, pathway, column_to_predict, limit):
        self.pathway = pathway
        self.column_to_predict = column_to_predict
        self.limit = limit
        self.laoddataset()
        self.dropnulls()
        self.definecriteria()
        self.labelencoding()
        self.traindatapreprocess()
        self.scaling()

    def laoddataset(self):
        self.wine = pd.read_csv(self.pathway)

    def dropnulls(self):
        self.wine = self.wine.dropna(axis=1)

    def definecriteria(self):
        min_value = min(self.wine[self.column_to_predict])-1
        max_value = max(self.wine[self.column_to_predict])+1
        bins = (min_value, self.limit, max_value)
        group_names = ["bad","good"]
        self.wine[self.column_to_predict] = pd.cut(self.wine[self.column_to_predict], bins=bins, labels=group_names)

    def labelencoding(self):
        label_quality = LabelEncoder()
        self.wine[self.column_to_predict] = label_quality.fit_transform(self.wine[self.column_to_predict])

    def traindatapreprocess(self):
        X = self.wine.drop(self.column_to_predict, axis=1)
        y = self.wine[self.column_to_predict]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

    def scaling(self):
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)

class RandomForestClassifying(WineQuality):
    def __init__(self, pathway, column_to_predict, limit, n_estimators = 100):
        super().__init__(pathway, column_to_predict, limit)
        self.n_estimators = n_estimators
        self.calculationRandomForestClassifying()
        self.reportsRFC()
        self.crossValidationRandomForest()

    def calculationRandomForestClassifying(self):
        self.rfc = RandomForestClassifier(self.n_estimators) # number of trees. Each tree will bring a prediction.
        self.rfc.fit(self.X_train, self.y_train)
        self.pred_rfc = self.rfc.predict(self.X_test)

class SupportVectorMachineClassifying(WineQuality):
    def __init__(self, pathway, column_to_predict, limit):
        super().__init__(pathway, column_to_predict, limit)
        self.calculationSupportVectorMachineClassifying()
        self.reportsSVM()
        self.crossValidationSVC()

    def calculationSupportVectorMachineClassifying(self):
        self.clf = svm.SVC()
        self.clf.fit(self.X_train, self.y_train)
        self.pred_clf = self.clf.predict(self.X_test)

class MultilayerPerceptronClassifying(WineQuality):
    def __init__(self, pathway, column_to_predict, limit, hidden_layer_sizes=(111,111,111), max_iter=5000):
        super().__init__(pathway, column_to_predict, limit)
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


# a = RandomForestClassifying(r"C:\\Users\\Monster PC\Downloads\winequality-red.csv", "quality", 6.5)
# b = SupportVectorMachineClassifying(r"C:\\Users\\Monster PC\Downloads\winequality-red.csv", "quality", 6.5)
# c = MultilayerPerceptronClassifying(r"C:\\Users\\Monster PC\Downloads\winequality-red.csv", "quality", 6.5)
