import dill as pkl
from shared_data import ProcessedData, PreprocessingInfo
import numpy as np

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
from matplotlib import pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import seaborn as sn

if __name__ == "__main__":
    with open("train.pkl", "rb") as fh:
        training_data: ProcessedData = pkl.load(fh)
    with open("test.pkl", "rb") as fh:
        test_data: ProcessedData = pkl.load(fh)

    if len(training_data.assignees.shape)==2:
        training_data.assignees = np.argmax(training_data.assignees, axis=1)
        test_data.assignees = np.argmax(test_data.assignees, axis=1)

    dummy = DummyClassifier()
    dummy.fit(training_data.bodies, training_data.assignees)
    dummy_score = dummy.score(test_data.bodies, test_data.assignees)

    #clf = dummy
    clf = MultinomialNB()
    #clf = SVC()
    #clf = KNeighborsClassifier(5)
    #clf = RandomForestClassifier()

    clf.fit(training_data.bodies, training_data.assignees)
    train_score = clf.score(training_data.bodies, training_data.assignees)
    test_score = clf.score(test_data.bodies, test_data.assignees)
    print(f"train score: {train_score}")
    print(f"test score: {test_score}")
    print(f"improvement over dummy: {test_score/dummy_score}")
    cf_train = confusion_matrix(test_data.assignees, clf.predict(test_data.bodies))
    tp = 0
    wrong = 0
    for i, row in enumerate(cf_train):
        for j, value in enumerate(row):
            if i==j:
                tp += value
            else:
                wrong += value
    print(f"TP: {tp}, wrong: {wrong}, classes: {len(cf_train)}")
    sn.heatmap(cf_train)
    plt.title(f"confusion matrix of {type(clf).__name__}")
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.show()