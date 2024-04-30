import dill as pkl
from model import ProcessedData, PreprocessingInfo
import numpy as np

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score

import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    with open("train.pkl", "rb") as fh:
        training_data: ProcessedData = pkl.load(fh)
    with open("test.pkl", "rb") as fh:
        test_data: ProcessedData = pkl.load(fh)

    #print(np.argmax(training_data.assignees, axis=1).shape)

    #import scipy
    #print(scipy.stats.mode(training_data.assignees))

    if len(training_data.assignees.shape)==2:
        training_data.assignees = np.argmax(training_data.assignees, axis=1)
        test_data.assignees = np.argmax(test_data.assignees, axis=1)

    #clf = SVC()
    #clf = DummyClassifier()
    clf = MultinomialNB()
    #clf = RandomForestClassifier()

    clf.fit(training_data.bodies, training_data.assignees)
    print(f"train score: {clf.score(training_data.bodies, training_data.assignees)}")
    print(f"test score: {clf.score(test_data.bodies, test_data.assignees)}")
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
    plt.show()

    

    import sys
    sys.exit("ONLY SINGLE CLASSIFIER FOR NOW")

    unique_assignees = np.unique(training_data.assignees)
    assignee_to_index = dict()
    for i, ass in enumerate(unique_assignees):
        assignee_to_index[ass]=i
    train_assignees = np.zeros((len(training_data.assignees), len(unique_assignees)+1))
    for i, assignee in enumerate(training_data.assignees):
        train_assignees[i,assignee_to_index[assignee]]=1
    test_assignees = np.zeros((len(test_data.assignees), len(unique_assignees)))
    for i, assignee in enumerate(test_data.assignees):
        if assignee in assignee_to_index:
            test_assignees[i,assignee_to_index[assignee]]=1
        else:
            test_assignees[i,-1]=1



    clfs = [MultinomialNB() for i in range(train_assignees.shape[1]-1)]
    for i, clf in enumerate(clfs):
        train_targets = (train_assignees[:,i]>=0.5)
        test_targets = (test_assignees[:,i]>=0.5)

        pos_train_indices = train_targets==1
        num_pos = pos_train_indices.sum()
        if num_pos==0:
            print("SKIP")
            continue

        repeat_amount = int(len(train_targets)/num_pos-2)
        print(repeat_amount)
        
        train_targets_strat = np.concatenate((train_targets, train_targets[pos_train_indices].repeat(repeat_amount, axis=0)))
        train_features_strat = np.concatenate((training_data.bodies, training_data.bodies[pos_train_indices].repeat(repeat_amount, axis=0)))

        #clf.fit(train_features_strat, train_targets_strat)
        clf.fit(training_data.bodies, train_targets)

        print(f"score on strat training data: {clf.score(train_features_strat, train_targets_strat)}")
        print(f"score on training data: {clf.score(training_data.bodies, train_targets)}")
        print(f"score on test data: {clf.score(test_data.bodies, test_targets)}")
        print(confusion_matrix(test_targets, clf.predict(test_data.bodies)))

    
    #clf = DecisionTreeClassifier()
    #train_targets = np.argmax(training_data.assignees, axis=1)
    #test_targets = np.argmax(test_data.assignees, axis=1)
    #clf.fit(training_data.bodies, train_targets)

    #print(f"score on training data: {clf.score(training_data.bodies, train_targets)}")
    #print(confusion_matrix(train_targets, clf.predict(training_data.bodies)))
    #print(f"score on test data: {clf.score(test_data.bodies, test_targets)}")
    #print(confusion_matrix(test_targets, clf.predict(test_data.bodies)))
    
    #
    print(training_data.preprocessing_info.word_list)