import os

train_data_input_path = os.path.join('data', 'train')
test_data_input_path = os.path.join('data', 'test')

reviews_train = []
for sentiment in ['pos', 'neg']:
    path = os.path.join(train_data_input_path, sentiment)
    for filename in os.listdir(path):
        for line in open(os.path.join(path, filename), 'r', encoding="utf8"):
            reviews_train.append(line.strip())


# test_data needs to be index by order for kaggle submission
reviews_test = []
path = test_data_input_path
for i in range(25000):
    for line in open(os.path.join(path, str(i)+".txt"), 'r', encoding="utf8"):
        reviews_test.append(line.strip())

for path in ('train.csv', 'test.csv'):
    with open(path, 'w', encoding="utf8") as f:
        for item in reviews_train:
            f.write("%s\n" % item)
