#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:28:24 2019

@author: svysali
"""
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def getSVM(X,target,c_initial=0.01,select_c=False):
    X_train, X_val, y_train, y_val = train_test_split(
            X, target, train_size = 0.75)
    max_accuracy = 0
    selected_c = c_initial
    if(select_c):
        for c in [0.01, 0.05, 0.25, 0.5, 1]:
            svm = LinearSVC(C=c)
            svm.fit(X_train, y_train)
            accuracy = accuracy_score(y_val, svm.predict(X_val))
            if(accuracy >= max_accuracy ):
                max_accuracy = accuracy
                selected_c = c        
            print("Accuracy for C={}: {}".format(c, accuracy))
        print("Selected c={}".format(selected_c)) 
    final_svm = LinearSVC(C=selected_c)
    final_svm.fit(X, target)
    print("Training Accuracy\t:\t{}".format(
                accuracy_score(target, final_svm.predict(X))
            ))
    return final_svm


def getLogisticRegression(X,target,c_initial=0.01,select_c=False):
    X_train, X_val, y_train, y_val = train_test_split(
            X, target, train_size = 0.75)
    max_accuracy = 0
    selected_c = c_initial
    if(select_c):
        for c in [0.01, 0.05, 0.25, 0.5, 1]:
            lr = LogisticRegression(C=c)
            lr.fit(X_train, y_train)
            accuracy = accuracy_score(y_val, lr.predict(X_val))
            if(accuracy >= max_accuracy):
                max_accuracy = accuracy
                selected_c = c        
            print("Validation Accuracy for C={}: {}".format(c, accuracy))
        print("Selected c={}".format(selected_c))
    final_log_reg = LogisticRegression(C=selected_c)
    final_log_reg.fit(X, target)
    print("Training Accuracy\t:\t{}".format(
                accuracy_score(target, final_log_reg.predict(X))
            ))
    return final_log_reg