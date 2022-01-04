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

class CreditCard():
    def __init__(self, pathway, column_to_predict):
        self.pathway = pathway
        self.column_to_predict = column_to_predict
        self.laoddataset()
        self.labelencoding()
        self.traindatapreprocess()
        self.scaling()

    def laoddataset(self):
        self.cc = pd.read_csv(self.pathway)

    def dropnulls(self):
        self.cc = self.cc.dropna(axis=1)

    def labelencoding(self):
        label_quality = LabelEncoder()
        self.cc[self.column_to_predict] = label_quality.fit_transform(self.cc[self.column_to_predict])

    def traindatapreprocess(self):
        X = self.cc.drop(self.column_to_predict, axis=1) #quality sütununu ayırdık ve geri kalanlar x tablosu oldu.
        y = self.cc[self.column_to_predict]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

    def scaling(self):
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train) # nested array dönüyor. fit_transform does the math to fit all the values
                                    # so that all the variables affect the model at the same level.
        self.X_test = sc.transform(self.X_test) # we just transformed because we didn't want to calculate
                              # mean and standard deviation again.

class RandomForestClassifying(CreditCard):
    def __init__(self, pathway, column_to_predict, n_estimators = 3):
        super().__init__(pathway, column_to_predict)
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

class SupportVectorMachineClassifying(CreditCard):
    def __init__(self, pathway, column_to_predict):
        super().__init__(pathway, column_to_predict)
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

class MultilayerPerceptronClassifying(CreditCard):
    def __init__(self, pathway, column_to_predict, hidden_layer_sizes=(1,1,1), max_iter=500):
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
    def __init__(self, pathway, column_to_predict):
        RandomForestClassifying.__init__(self, pathway, column_to_predict, n_estimators = 3)
        SupportVectorMachineClassifying.__init__(self, pathway, column_to_predict)
        MultilayerPerceptronClassifying.__init__(self, pathway, column_to_predict, hidden_layer_sizes=(11,11,11), max_iter=1000)
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
    def __init__(self, pathway, column_to_predict):
        RandomForestClassifying.__init__(self, pathway, column_to_predict, n_estimators = 200)
        SupportVectorMachineClassifying.__init__(self, pathway, column_to_predict)
        MultilayerPerceptronClassifying.__init__(self, pathway, column_to_predict, hidden_layer_sizes=(2,2,2), max_iter=50)
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


# a = RandomForestClassifying(r"C:\\Users\\Monster PC\Downloads\load_breast_cancer.csv", "diagnosis")
# b = SupportVectorMachineClassifying(r"C:\\Users\\Monster PC\Downloads\load_breast_cancer.csv", "diagnosis")
# c = MultilayerPerceptronClassifying(r"C:\\Users\\Monster PC\Downloads\load_breast_cancer.csv", "diagnosis")
# d = Reports(r"C:\Users\Monster PC\Desktop\UCI_Credit_Card.csv", "default.payment.next.month")
e = CrossValidation(r"C:\Users\Monster PC\Desktop\UCI_Credit_Card.csv", "default.payment.next.month")
