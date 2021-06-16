from csv import reader
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier



def load_dataset(dataset_path):
    attribute = []
    res = []
    with open(dataset_path, 'r') as file:
        csv_reader = reader(file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            attribute.append(row[:-1])
            res.append(row[-1])
    return attribute, res

attribute, res = load_dataset('glass.csv')

X_train, X_test, y_train, y_test = train_test_split(attribute, res, test_size=0.6)

svm = SVC()

svm.fit(X_train, y_train)




###  Adaboost 集成分类器
clf = AdaBoostClassifier(n_estimators=200, learning_rate=0.60)
clf.fit(X_train, y_train)
print("train:",clf.score(X_train, y_train))
print("test：",clf.score(X_test, y_test))


