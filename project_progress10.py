

# Here instructed, I combined all projects in one code structure.
# According to the chosen path, the system can now be applied to all datasets.

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

class Preprocess():
    def __init__(self, pathway, column_to_predict, n_estimators = 100, hidden_layer_sizes = (11,11,11), max_iter=200):
        self.pathway = pathway
        self.column_to_predict = column_to_predict
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        self.laoddataset()
        self.dropnulls()
        self.labelencoding()
        self.traindatapreprocess()
        self.scaling()

    def laoddataset(self):
        self.data = pd.read_csv(self.pathway)

    def dropnulls(self):
        self.data = self.data.dropna(axis=1)
        if self.column_to_predict == "FoodGroup":
            self.data = self.data.drop(["ID", "ShortDescrip", "Descrip"], axis = 1)
        pass

    def labelencoding(self):
        label_quality = LabelEncoder()
        self.data[self.column_to_predict] = label_quality.fit_transform(self.data[self.column_to_predict])

    def traindatapreprocess(self):
        X = self.data.drop(self.column_to_predict, axis=1) #quality sütununu ayırdık ve geri kalanlar x tablosu oldu.
        y = self.data[self.column_to_predict]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

    def scaling(self):
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train) # nested array dönüyor. fit_transform does the math to fit all the values
                                    # so that all the variables affect the model at the same level.
        self.X_test = sc.transform(self.X_test) # we just transformed because we didn't want to calculate

class RandomForestClassifier(Preprocess):
    def __init__(self, pathway, column_to_predict, n_estimators, hidden_layer_sizes, max_iter):
        super().__init__(pathway, column_to_predict, n_estimators, hidden_layer_sizes, max_iter)
        self.n_estimators = n_estimators
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.calculationRandomForestClassifier()
        self.reportsRFC()
        self.crossValidationRandomForest()

    def calculationRandomForestClassifier(self):
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

class SupportVectorMachineClassifier(Preprocess):
    def __init__(self, pathway, column_to_predict, n_estimators, hidden_layer_sizes, max_iter):
        super().__init__(pathway, column_to_predict, n_estimators, hidden_layer_sizes, max_iter)
        self.calculationSupportVectorMachineClassifier()
        self.reportsSVM()
        self.crossValidationSVC()

    def calculationSupportVectorMachineClassifier(self):
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

class MultilayerPerceptronClassifier(Preprocess):
    def __init__(self, pathway, column_to_predict, n_estimators, hidden_layer_sizes, max_iter):
        super().__init__(pathway, column_to_predict, n_estimators, hidden_layer_sizes, max_iter)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.calculationMultilayerPerceptronClassifier()
        self.reportsMLPC()
        self.crossValidationMLPC()

    def calculationMultilayerPerceptronClassifier(self):
        self.mlpc = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter)
#                     #3 hidden layers and size of each are 11
#                     #maximum iterations is 500. It means, it will go through the data 500 times.
        self.mlpc.fit(self.X_train, self.y_train)
        self.pred_mlpc = self.mlpc.predict(self.X_test)

    def reportsMLPC(self):
        self.crmlpc= classification_report(self.y_test, self.pred_mlpc)
        self.cmmlpc = confusion_matrix(self.y_test, self.pred_mlpc)

    def crossValidationMLPC(self):
        crossmlpc = self.calculationMultilayerPerceptronClassifier()
        total = 0
        validationlistmlpc = cross_val_score(self.mlpc, self.X_train, self.y_train)
        for i in validationlistmlpc:
            total = total + i
        self.avg_mlpc = round(total/len(validationlistmlpc),3)

class CrossValidation(RandomForestClassifier, SupportVectorMachineClassifier, MultilayerPerceptronClassifier):
    def __init__(self, pathway, column_to_predict, n_estimators=100, hidden_layer_sizes=(50,50,50), max_iter=1000):
        RandomForestClassifier.__init__(self, pathway, column_to_predict, n_estimators, hidden_layer_sizes, max_iter)
        SupportVectorMachineClassifier.__init__(self, pathway, column_to_predict, n_estimators, hidden_layer_sizes, max_iter)
        MultilayerPerceptronClassifier.__init__(self, pathway, column_to_predict, n_estimators, hidden_layer_sizes, max_iter)
        self.n_estimators = n_estimators
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.crossValidationComparison()
        self.showchart()

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

    def showchart(self):
        models = list(self.my_dict.keys())
        values = list(self.my_dict.values())
        fig = plt.figure(figsize = (10, 5))
        plt.bar(models, values, color =['olive', "lightskyblue", "lightcoral"],  width = 0.4)
        for i in range(len(models)):
            plt.text(i,values[i]/2, values[i], ha = 'center')
        plt.xlabel("MODELS")
        plt.ylabel("Accuracy")
        plt.title("Comparison of Models")
        plt.show()


# e = CrossValidation(r"C:\Users\Monster PC\Desktop\UCI_Credit_Card.csv", "default.payment.next.month")
# d = CrossValidation(r"C:\Users\Monster PC\Desktop\Datasets\nndb_flat.csv", "FoodGroup")
