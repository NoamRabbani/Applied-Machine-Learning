from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os
import re

"""
Step 2: Read into Python
"""


train_data_dir = os.path.join('.', 'data', 'train')
train_data_dir = os.path.join('.', 'data', 'train')
test_data_dir = os.path.join('.', 'data', 'test')

subset = 1000
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
print(X[5, 5])
