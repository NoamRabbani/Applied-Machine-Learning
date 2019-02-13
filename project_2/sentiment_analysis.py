from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
import pdb
import numpy as np
import pandas as pd
import os
import re
import csv

"""
Step 2: Read into Python
"""


train_data_dir = os.path.join('.', 'data', 'train')
train_data_dir = os.path.join('.', 'data', 'train')
test_data_dir = os.path.join('.', 'data', 'test')

subset = 25000
reviews_train = []
for sentiment in ['pos', 'neg']:
    path = os.path.join(train_data_dir, sentiment)
    for filename in os.listdir(path):
        if subset/2 < 0:
            break
        for line in open(os.path.join(path, filename), 'r', encoding="utf8"):
            reviews_train.append(line.strip())
            subset -= 1

subset = 1000
reviews_test = []
path = test_data_dir
for filename in os.listdir(path):
    if subset < 0:
        break
    for line in open(os.path.join(path, filename), 'r', encoding="utf8"):
        reviews_test.append(line.strip())
        subset -= 1

print(reviews_train[5])

"""
Step 3: Clean and Preprocess
"""
# this part doesn't seem to be working on python 3


REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")  # noqa
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")  # noqa
NO_SPACE = ""
SPACE = " "


def preprocess_reviews(reviews):

    reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower())
               for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]

    return reviews


reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)


reviews_train_clean[5]


"""
Vectorization
"""

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
cv = CountVectorizer(binary=True)
cv.fit(reviews_train)
X = cv.transform(reviews_train)
X_test = cv.transform(reviews_test)

# X : sparse matrix, [n_samples, n_features]
print(X.shape)
print(X[5, 5])

target = [1 if i < 12500 else 0 for i in range(25000)]

X_train, X_val, y_train, y_val = train_test_split(
            X, target, train_size = 0.75, test_size =0.25)
bnb = BernoulliNB()
print('size train x: '+str(X_train.shape))
print('size train y: '+str(len(y_train)))

print('size val x: '+str(X_val.shape))
print('size val y: '+str(len(y_val)))
bnb.fit( X_train,y_train)

y_pred = bnb.predict(X_val)

print('predicted by scikit')
print(y_pred)

class MyBernoulliNB(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        count_sample = X.shape[0]
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.class_log_prior_ = [np.log(len(i) / count_sample) for i in separated]
        count = np.array([np.array(i).sum(axis=0) for i in separated])
        count = np.concatenate([item.toarray()+self.alpha for item in count])
        smoothing = 2 * self.alpha
        n_doc = np.array([len(i) + smoothing for i in separated])
        print(n_doc)
        self.feature_prob_ = count / n_doc[np.newaxis].T
        return self
    def predict_log_proba(self, X):
        print('in log proba')
        return [(np.log(self.feature_prob_) * x.toarray()[0] + \
                 np.log(1 - self.feature_prob_) * np.abs(x.toarray()[0] - 1)
                 ).sum(axis=1) + self.class_log_prior_ for x in X]
    def predict(self, X):
        print('in predict')
        return np.argmax(self.predict_log_proba(X), axis=1)
nb = MyBernoulliNB(alpha=1).fit(X_train , y_train)

print('predicted by my implementation')
my_y_pred=nb.predict(X_val)
print(my_y_pred)
resultfile = open("output.csv",'w')

for indx in range(len(y_pred)):
    resultfile.write(str(y_pred[indx]) + ", "+str(my_y_pred[indx])+"\n")
resultfile.close()
