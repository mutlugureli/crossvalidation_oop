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
    def __init__(self, pathway, column_to_predict, limit, testing_list=[]):
        self.pathway = pathway
        self.column_to_predict = column_to_predict
        self.limit = limit
        self.scaling()
        self.randomForestClassifying()
        self.supportVectorMachineClassifying()
        self.multilayerPerceptronClassifying()

    def laoddataset(self):
        wine = pd.read_csv(self.pathway)
        return wine

    def definecriteria(self):
        wineloaded = self.laoddataset()
        min_value = min(wineloaded[self.column_to_predict])
        max_value = max(wineloaded[self.column_to_predict])
        bins = (min_value, self.limit, max_value)
        group_names = ["bad","good"]
        wineloaded[self.column_to_predict] = pd.cut(wineloaded[self.column_to_predict], bins=bins, labels=group_names)
        return wineloaded

    def labelencoding(self):
        label_quality = LabelEncoder()
        wineencoded = self.definecriteria()
        wineencoded[self.column_to_predict] = label_quality.fit_transform(wineencoded[self.column_to_predict])
        return wineencoded

    def trainDataPreprocess(self):
        winelabelled = self.labelencoding()
        X = winelabelled.drop(self.column_to_predict, axis=1)
        y = winelabelled[self.column_to_predict]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        traindatalist = [X_train, X_test, y_train, y_test]
        return traindatalist

    def scaling(self):
        listtobescaled = self.traindatapreprocess()
        sc = StandardScaler()
        X_train = sc.fit_transform(listtobescaled[0])
        X_test = sc.transform(listtobescaled[1])
        scaledlist = [X_train, X_test, listtobescaled[2], listtobescaled[3]]
        return scaledlist

    def randomForestClassifying(self):
        winescaled = self.scaling()
        rfc = RandomForestClassifier(n_estimators=350)
        rfc.fit(winescaled[0], winescaled[2])
        pred_rfc = rfc.predict(winescaled[1])
        print(classification_report(winescaled[3], pred_rfc))
        print(confusion_matrix(winescaled[3], pred_rfc))
