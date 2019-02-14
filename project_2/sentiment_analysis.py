from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from feature_extraction import remove_stop_words,get_lemmatized_text,get_stemmed_text
from classifiers import getSVM, getLogisticRegression
import pdb
import numpy as np
import sys
import os
import re
import csv

"""
Step 2: Read into Python
"""


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

subset = 25000
reviews_test = []
path = test_data_dir

#test_data needs to be index by order for kaggle submission
for i in range(25000):
    if subset < 0:
        break
    for line in open(os.path.join(path, str(i)+".txt"), 'r', encoding="utf8"):
        reviews_test.append(line.strip())
        subset -= 1

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

def write_output_csv(output_file,y_pred):
    with open(output_file, 'w') as csvfile:
        fieldnames = ['Id', 'Category']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(y_pred)):
            writer.writerow({'Id': index, 'Category': y_pred[index]})


reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)



"""
Data set with no processing
"""

target = [1 if i < 12500 else 0 for i in range(25000)]

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
#only count
#cv = CountVectorizer(binary=True)
#cv.fit(reviews_train)
#X = cv.transform(reviews_train)
#X_test = cv.transform(reviews_test)

## X : sparse matrix, [n_samples, n_features]
##print(X.shape)
##print(X[5, 5])
#
#X_train, X_val, y_train, y_val = train_test_split(
#            X, target, train_size = 0.75, test_size =0.25)

"""
naive bayes from scikit - only for comparing our implementation
"""

#bnb = BernoulliNB()
#print('size train x: '+str(X_train.shape))
#print('size train y: '+str(len(y_train)))
#
#print('size val x: '+str(X_val.shape))
#print('size val y: '+str(len(y_val)))
#bnb.fit( X_train,y_train)
#
#y_pred = bnb.predict(X_val)
#
#print('predicted by scikit')
#print(y_pred)

"""
naive bayes implementation
"""

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


"""
Task 1 : Running Naive Bayes from scratch
"""

#nb = MyBernoulliNB(alpha=1).fit(X_train , y_train)

#print('predicted by my implementation')
#my_y_pred=nb.predict(X_val)
#print(my_y_pred)
#resultfile = open("output.csv",'w')
#
#for indx in range(len(y_pred)):
#    resultfile.write(str(y_pred[indx]) + ", "+str(my_y_pred[indx])+"\n")
#resultfile.close()


"""
Task 2 : Running Two Different classifiers
"""

print("############################################")
print("# 2.1 Support Vector Machine with bigrams  #")
print("############################################")
#unigram and bigram count
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit(reviews_train_clean)
X_bigram = ngram_vectorizer.transform(reviews_train_clean)
model = getSVM(X_bigram,target)

print("############################################")
print("# 2.2 Logistic Regression with bigrams     #")
print("############################################")
model = getLogisticRegression(X_bigram,target)

"""
Task 3 : Feature Extraction Pipelines
"""
#print("############################################")
#print("# 3.1 SVM with No all stop words removed   #")
#print("############################################")
#no_stop_words_train = remove_stop_words(reviews_train_clean)
#no_stop_words_test = remove_stop_words(reviews_test_clean)
#
#ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
#ngram_vectorizer.fit(reviews_train_clean)
#X_no_stop_words = ngram_vectorizer.transform(no_stop_words_train)
#X_test_no_stop_words = ngram_vectorizer.transform(no_stop_words_test)
#X_train, X_val, y_train, y_val = train_test_split(
#    X_no_stop_words, target, train_size = 0.75
#)
#max_accuracy_c_svm = 0
#selected_c = 0.01
#check_for_c = False
#if(check_for_c):
#    for c in [0.01, 0.05, 0.25, 0.5, 1]:
#        svm = LinearSVC(C=c)
#        svm.fit(X_train, y_train)
#        accuracy = accuracy_score(y_val, svm.predict(X_val))
#        if(accuracy >= max_accuracy_c_svm ):
#            max_accuracy_c_svm = accuracy
#            selected_c = c        
#        print("Accuracy for C={}: {}".format(c, accuracy))
#print("Selected c={}".format(selected_c)) 

#lemmatized_reviews = get_lemmatized_text(reviews_train_clean)

"""
Task 4 : Validation pipeline
"""





"""
Task 5 : Final model
"""
print("############################################")
print("# Submitted model performances             #")
print("############################################")
# 14/02 1/2 - Trigrams + some stop words removed. 
stop_words = ['in', 'of', 'at', 'a', 'the']
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)
ngram_vectorizer.fit(reviews_train_clean)
X = ngram_vectorizer.transform(reviews_train_clean)
X_test = ngram_vectorizer.transform(reviews_test_clean)
final_model_1 = getSVM(X,target,0.01)


"""
Run predict function of whichever model you want to run and 
write results to csv to submit to kaggle
"""
#Y_pred = final_model_1.predict(X_test_final)
#write_output_csv("results_final.csv",Y_pred)