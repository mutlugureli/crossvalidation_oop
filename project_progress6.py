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

    def reportsRFC(self):
        self.crrfc = classification_report(self.y_test, self.pred_rfc)
        self.cmrfc = confusion_matrix(self.y_test, self.pred_rfc)


    def crossValidationRandomForest(self):
        validationlistrfc = cross_val_score(self.rfc, self.X_train, self.y_train)
        total = 0
        for i in validationlistrfc:
            total = total + i
        self.avg_rfc = round(total/len(validationlistrfc),3)

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

    def reportsSVM(self):
        self.crsvm= classification_report(self.y_test, self.pred_clf)
        self.cmsvm = confusion_matrix(self.y_test, self.pred_clf)

    def crossValidationSVC(self):
        total = 0
        validationlistsvc = cross_val_score(self.clf, self.X_train, self.y_train)
        for i in validationlistsvc:
            total = total + i
        self.avg_svc = round(total/len(validationlistsvc),3)

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

    def reportsMLPC(self):
        self.crmlpc= classification_report(self.y_test, self.pred_mlpc)
        self.cmmlpc = confusion_matrix(self.y_test, self.pred_mlpc)

    def crossValidationMLPC(self):
        crossmlpc = self.calculationMultilayerPerceptronClassifying()
        total = 0
        validationlistmlpc = cross_val_score(self.mlpc, self.X_train, self.y_train)
        for i in validationlistmlpc:
            total = total + i
        self.avg_mlpc = round(total/len(validationlistmlpc),3)

class Reports(RandomForestClassifying, SupportVectorMachineClassifying, MultilayerPerceptronClassifying):
    def __init__(self, pathway, column_to_predict, limit):
        RandomForestClassifying.__init__(self, pathway, column_to_predict, limit, n_estimators = 200)
        SupportVectorMachineClassifying.__init__(self, pathway, column_to_predict, limit)
        MultilayerPerceptronClassifying.__init__(self, pathway, column_to_predict, limit, hidden_layer_sizes=(11,11,11), max_iter=1000)
        self.reportPrint()

    def reportPrint(self):
        cr_dic = {
            "Random Forest Classifier classification report": self.crrfc,
            "Support Vector Machine classification report: ": self.crsvm,
            "Multilayer Perceptron Classifier classification report: ": self.crmlpc
        }
        for model, value in cr_dic.items():
            print(model)
            print(value)
        print("\n\n")
        cm_dic = {
            "Random Forest Classifier confusion matrix:": self.cmrfc,
            "Support Vector Machine confusion matrix: ": self.cmsvm,
            "Multilayer Perceptron Classifier confusion matrix: ": self.cmmlpc
        }
        for model, value in cm_dic.items():
            print(model)
            print(value)
        print("\n\n")

class CrossValidation(RandomForestClassifying, SupportVectorMachineClassifying, MultilayerPerceptronClassifying):
    def __init__(self, pathway, column_to_predict, limit):
        RandomForestClassifying.__init__(self, pathway, column_to_predict, limit, n_estimators = 400)
        SupportVectorMachineClassifying.__init__(self, pathway, column_to_predict, limit)
        MultilayerPerceptronClassifying.__init__(self, pathway, column_to_predict, limit, hidden_layer_sizes=(2,2,2), max_iter=5000)
        self.crossValidationComparison()

    def crossValidationComparison(self):
        self.my_dict = {
            "Random Forest Classifier" : float(self.avg_rfc),
            "Support Vector Machine" : float(self.avg_svc),
            "Multilayer Perceptron Classifier": float(self.avg_mlpc)
        }
        best_pred = max(self.avg_mlpc, self.avg_svc, self.avg_rfc)
        best_model = list(self.my_dict.keys())[list(self.my_dict.values()).index(best_pred)]

        print("The cross validation results are:")
        for key, value in self.my_dict.items():
            print(key+": " +str(value))
        print("\n")
        print ("Best working model for this dataset is: {}".format(best_model))
        print("The accuracy of {} model: {}".format(best_model, best_pred))


# a = RandomForestClassifying(r"C:\\Users\\Monster PC\Downloads\winequality-red.csv", "quality", 6.5)
# b = SupportVectorMachineClassifying(r"C:\\Users\\Monster PC\Downloads\winequality-red.csv", "quality", 6.5)
# c = MultilayerPerceptronClassifying(r"C:\\Users\\Monster PC\Downloads\winequality-red.csv", "quality", 6.5)
# d = Reports(r"C:\\Users\\Monster PC\Downloads\winequality-red.csv", "quality", 7)
cross_obj = CrossValidation(r"C:\\Users\\Monster PC\Downloads\winequality-red.csv", "quality", 6.5)
