from feature_extraction import remove_stop_words,get_lemmatized_text,get_stemmed_text
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from classifiers import getSVM, getLogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os
import re
import csv
import pdb

"""
Define functions
"""

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
        self.feature_prob_ = count / n_doc[np.newaxis].T
        return self
    def predict_log_proba(self, X):
        return [(np.log(self.feature_prob_) * x.toarray()[0] + \
                 np.log(1 - self.feature_prob_) * np.abs(x.toarray()[0] - 1)
                 ).sum(axis=1) + self.class_log_prior_ for x in X]
    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)



"""
Step 1: Read into Python
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

for i in range(25000):
    if subset < 0:
        break
    for line in open(os.path.join(path, str(i)+".txt"), 'r', encoding="utf8"):
        reviews_test.append(line.strip())
        subset -= 1

"""
Step 2: Clean and Preprocess
"""


REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")  # noqa
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")  # noqa
NO_SPACE = ""
SPACE = " "



reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)

no_sw_train = remove_stop_words(reviews_train_clean) 
#no_sw_test = remove_stop_words(reviews_test_clean)

lemmatized_train = get_lemmatized_text(reviews_train_clean) 
#lemmatized_test = get_lemmatized_text(reviews_test_clean)

stemmed_train = get_stemmed_text(reviews_train_clean)


target = [1 if i < 12500 else 0 for i in range(25000)]

cv = CountVectorizer(binary=True)
cv.fit(reviews_train)
X = cv.transform(reviews_train)

X_train, X_val, y_train, y_val = train_test_split(
            X, target, train_size = 0.75, test_size =0.25)

"""
Step 3: verify NB classifier implementation
"""
print('****************************')
print('*NB Task1: verification*****')
print('****************************')
print('predicted by scikit-learn')
bnb = BernoulliNB()
bnb.fit( X_train,y_train)
y_pred = bnb.predict(X_val)

print('predicted by my implementation')
nb = MyBernoulliNB(alpha=1).fit(X_train , y_train)
my_y_pred=nb.predict(X_val)

sciaccuracy= accuracy_score(y_val, bnb.predict(X_val))
myaccuracy = accuracy_score(y_val, nb.predict(X_val))
print("scikit accuracy: "+str(sciaccuracy)+"; my accuracy: "+str(myaccuracy))


print('****************************')
print('*NB Task2: alpha study******')
print('****************************')
for al in [0,0.2,0.5,1,2,3,4,5]:
   nb = MyBernoulliNB(alpha=al).fit(X_train , y_train)
   my_y_pred=nb.predict(X_val)
   myaccuracy = accuracy_score(y_val, nb.predict(X_val))
   print("alpha: "+str(al)+"; accuracy: "+str(myaccuracy))

print('****************************')
print('*NB Task3: vocabulary study*')
print('****************************')
for partial in [0.1,0.2,0.5,0.8,1]:
    cv = CountVectorizer(binary=True,max_df=partial)
    cv.fit(reviews_train)
    X = cv.transform(reviews_train)
    X_train, X_val, y_train, y_val = train_test_split(
                X, target, train_size = 0.75, test_size =0.25)
    nb = MyBernoulliNB(alpha=1).fit(X_train , y_train)
    my_y_pred=nb.predict(X_val)
    myaccuracy = accuracy_score(y_val, nb.predict(X_val))
    print(str(partial)+" of the vocabulary used, accuracy: "+str(myaccuracy))


print('****************************')
print('*NB Task4: pre-processing **')
print('****************************')
cv.fit(no_sw_train)
X = cv.transform(no_sw_train)

X_train, X_val, y_train, y_val = train_test_split(
                    X, target, train_size = 0.75, test_size =0.25)
nb = MyBernoulliNB(alpha=1).fit(X_train , y_train)

print('removed stop words')
my_y_pred=nb.predict(X_val)
myaccuracy = accuracy_score(y_val, nb.predict(X_val))
print("accuracy: "+str(myaccuracy))

cv.fit(lemmatized_train)
X = cv.transform(lemmatized_train)

X_train, X_val, y_train, y_val = train_test_split(
                            X, target, train_size = 0.75, test_size =0.25)
nb = MyBernoulliNB(alpha=1).fit(X_train , y_train)

print('lemmatized')
my_y_pred=nb.predict(X_val)
myaccuracy = accuracy_score(y_val, nb.predict(X_val))
print("accuracy: "+str(myaccuracy))

cv.fit(stemmed_train)
X = cv.transform(stemmed_train)

X_train, X_val, y_train, y_val = train_test_split(
                                    X, target, train_size = 0.75, test_size =0.25
                                    )
nb = MyBernoulliNB(alpha=1).fit(X_train , y_train)

print('stemmed')
my_y_pred=nb.predict(X_val)
myaccuracy = accuracy_score(y_val, nb.predict(X_val))
print("accuracy: "+str(myaccuracy))

