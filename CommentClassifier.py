import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, auc, roc_curve
import matplotlib.pylab as plt
import matplotlib.patches as patches

def read_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        result = [line.strip() for line in lines]
    return result

def fill_concern(data):
    result = []
    for i in range(len(data)):
        result.append(1)
    return result

def fill_others(data):
    result = []
    for i in range(len(data)):
        result.append(0)
    return result

concerns = np.array(read_file('concerns.txt'))
others = np.array(read_file('others.txt'))

# 1 = concern while  0 = others
x = np.append(concerns,others)
y = np.append(fill_concern(concerns), fill_others(others))

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

mnb = MultinomialNB()
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)
svc = SVC(random_state=42)

classifiers = [mnb, dt, rf, svc]
classifier_names = ["Multinomial NB", "Decision Tree", "Random Forest", "SVC"]
i=0
for classifier in classifiers:

    fold_accuracy, fold_precision, fold_recall, fold_f1 = [], [], [], []

    for train_index, test_index in cv.split(x, y):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        classifier.fit(X_train_vec, y_train)
        y_preds = classifier.predict(X_test_vec)

        fold_accuracy.append(accuracy_score(y_test, y_preds))
        fold_precision.append(precision_score(y_test, y_preds))
        fold_recall.append(recall_score(y_test, y_preds))
        fold_f1.append(f1_score(y_test, y_preds))

    print(classifier_names[i])
    print("Accuracy: ", sum(fold_accuracy)/10)
    print("Precision: ",sum(fold_precision)/10)
    print("Recall: ", sum(fold_recall)/10)
    print("F1-Score: ", sum(fold_f1)/10, "\n")

    i += 1
