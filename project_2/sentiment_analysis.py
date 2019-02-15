from feature_extraction import remove_stop_words,get_lemmatized_text
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from classifiers import getSVM, getLogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
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

no_sw_train = remove_stop_words(reviews_train_clean) 
no_sw_test = remove_stop_words(reviews_test_clean)

lemmatized_train = get_lemmatized_text(reviews_train_clean) 
lemmatized_test = get_lemmatized_text(reviews_test_clean)
"""
Data set with no processing
"""

target = [1 if i < 12500 else 0 for i in range(25000)]

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
#only count
cv = CountVectorizer(binary=True)
cv.fit(reviews_train)
X = cv.transform(reviews_train)
X_test = cv.transform(reviews_test)

X_train, X_val, y_train, y_val = train_test_split(
            X, target, train_size = 0.75, test_size =0.25)

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

print("##########################################################")
print("# 2.1 Classifier : Support Vector Machine with bigrams   #")
print("##########################################################")
#unigram and bigram count
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit(reviews_train_clean)
X_bigram = ngram_vectorizer.transform(reviews_train_clean)
model = getSVM(X_bigram,target)
#
print("##########################################################")
print("# 2.2 Classifier : Logistic Regression with bigrams      #")
print("##########################################################")
model = getLogisticRegression(X_bigram,target)

"""
Task 3 : Feature Extraction Pipelines
"""
print("############################################")
print("# 3.1 Binary Occurences                    #")
print("############################################")
bin_vectorizer = CountVectorizer(binary=True)
bin_vectorizer.fit(reviews_train_clean)
X_bin = bin_vectorizer.transform(reviews_train_clean)
model = getSVM(X_bin,target)

print("############################################")
print("# 3.2 TF-IDF                               #")
print("############################################")
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(reviews_train_clean)
X_tfidf = tfidf_vectorizer.transform(reviews_train_clean)
model = getSVM(X_tfidf,target,select_c=True)

"""
Task 5 : Final model
"""
print("############################################")
print("# Submitted model performances             #")
print("############################################")
# 14/02 1/2 - Trigrams + binary
stop_words = ['in', 'of', 'at', 'a', 'the']
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)
ngram_vectorizer.fit(reviews_train_clean)
X = ngram_vectorizer.transform(reviews_train_clean) 
X_test = ngram_vectorizer.transform(reviews_test_clean)
final_model_1 = getSVM(X,target,0.01)


# submit another day
stop_words = ['in', 'of', 'at', 'a', 'the']
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)
ngram_vectorizer.fit(lemmatized_train)
X = ngram_vectorizer.transform(lemmatized_train) 
X_test = ngram_vectorizer.transform(lemmatized_test)
final_model_1 = getSVM(X,target,0.01)
#Y_pred = final_model_1.predict(X_test)
#write_output_csv("results_lemmatized.csv",Y_pred)
