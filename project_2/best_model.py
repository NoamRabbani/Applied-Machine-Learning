from feature_extraction import remove_stop_words,get_lemmatized_text,get_stemmed_text
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from classifiers import getSVM, getLogisticRegression, getSGD
from sklearn.model_selection import train_test_split
import numpy as np
import os
import re
import csv
from scipy.sparse import hstack
import pdb

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


stemmed_train = get_stemmed_text(reviews_train_clean)
stemmed_test = get_stemmed_text(reviews_test_clean)

target = [1 if i < 12500 else 0 for i in range(25000)]



"""
Final model
"""
print("############################################")
print("# Submitted model performances             #")
print("############################################")
stop_words = ['in', 'of', 'at', 'a', 'the']
mystopwords = open("mystopwords.txt", 'r' , encoding="ISO-8859-1").read()
mystopwords = mystopwords.split("\n")

ngram_tfidf_vectorizer=TfidfVectorizer(binary=True, ngram_range=(1, 3), max_df=0.2,min_df=3, stop_words=stop_words)
ngram_tfidf_vectorizer.fit(stemmed_train)
X_ngram_tfidf = ngram_tfidf_vectorizer.transform(stemmed_train)
final_model_1 = getSVM(X_ngram_tfidf,target,c_initial=0.5)
feature_to_coef={word: coef for word, coef in zip(ngram_tfidf_vectorizer.get_feature_names(),final_model_1.coef_[0])}

for best_positive in sorted(
    feature_to_coef.items(),key=lambda x:x[1],reverse=True)[:10]: 
    print(best_positive)

 for best_negative in sorted(
     feature_to_coef.items(),key=lambda x:x[1])[:10]: 
     print(best_negative)

