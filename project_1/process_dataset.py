import json
import os
import re
import numpy as np
import pandas as pd
from textstat.textstat import textstat
from textblob import TextBlob
from collections import Counter


class DataProcessor():
    """ Handles the processing of the dataset containing reddit comments

    Attributes:
        dataset: list of dicts containing reddit comments and their attributes
    """

    def __init__(self, input_data_path):
        """Inits DataProcessor a dataset containing reddit comments"""
        self.input_data_path = input_data_path
        with open(input_data_path) as fp:
            self.dataset = json.load(fp)
        for idx in range(len(self.dataset)):
            self.dataset[idx]['text'] = self.process_text(
                self.dataset[idx]['text'])
        self.top_160_words = self.get_160_most_frequent_words(
            self.dataset[:10000])

    def generate_df_features(self, dataset, top_words,
                             features=[0, 0, 0, 0, 0, 0, 0, 0, 0],
                             interaction=None):
        bias = self.generate_df_bias(dataset)

        children = self.generate_df_children(dataset)
        controversiality = self.generate_df_controversiality(dataset)
        is_root = self.generate_df_is_root(dataset)

        df_features = pd.concat(
            [bias, children, controversiality, is_root], axis=1)

        if top_words > 0:
            top_words = self.generate_df_most_frequent_words(
                dataset, self.top_160_words[:top_words])
            df_features = pd.concat([df_features, top_words], axis=1)

        if features[0] > 1:
            children = self.generate_df_children(dataset, features[0])
            df_features = pd.concat([df_features, children], axis=1)
        if features[1] > 1:
            controversiality = self.generate_df_controversiality(
                dataset, features[1])
            df_features = pd.concat([df_features, controversiality], axis=1)
        if features[2] > 1:
            is_root = self.generate_df_is_root(dataset, features[2])
            df_features = pd.concat([df_features, is_root], axis=1)

        if features[3] > 0:
            word_count = self.generate_df_word_count(
                dataset, features[3])
            df_features = pd.concat([df_features, word_count], axis=1)
        if features[4] > 0:
            word_length = self.generate_df_avg_word_length(
                dataset, features[4])
            df_features = pd.concat([df_features, word_length], axis=1)
        if features[5] > 0:
            nb_word_per_sent = self.generate_df_nb_words_per_sent(
                dataset, features[5])
            df_features = pd.concat([df_features, nb_word_per_sent], axis=1)
        if features[6] > 0:
            sentiment = self.generate_df_sentiment(dataset, features[6])
            df_features = pd.concat([df_features, sentiment], axis=1)
        if features[7] > 0:
            readability = self.generate_df_readability(
                dataset, features[7])
            df_features = pd.concat([df_features, readability], axis=1)
        if features[8] > 0:
            sentiment = self.generate_df_sentiment(dataset)
            sentiment_square = self.generate_interact(sentiment,'sentiment',sentiment,'sentiment')
            nb_word_per_sent = self.generate_df_nb_words_per_sent(dataset)
            nb_word_per_sent_sentiment2 = self.generate_interact(
                sentiment_square,'sentiment*sentiment',nb_word_per_sent,'nb_word_per_sent')
            df_features = pd.concat([df_features, nb_word_per_sent_sentiment2], axis=1)

        if interaction is not None:
            interaction = self.generate_interact(
                df_features, interaction[0], df_features, interaction[1])
            df_features = pd.concat([df_features, interaction], axis=1)

        df_output = self.generate_df_popularity_score(dataset)
        return df_features, df_output

    # extra features begin here

    def generate_df_word_count(self, dataset, exponent=1):
        """ Generates a 10000x1 df of the word count for each comment

        Args:
            dataset: list of dicts containing reddit comments and
            their attributes

        Returns:
            df: 10000x1 pandas df that contains the word count
            for every training instance
        """
        if exponent == 1:
            column = ['word_count']
        elif exponent > 1:
            column = ['word_count**' + str(exponent)]

        rows = []
        for data_point in dataset:
            for info_name, info_value in data_point.items():
                if info_name == 'text':
                    word_count = info_value.size
                    if exponent > 1:
                        word_count = word_count ** exponent
            rows.append(word_count)

        df = pd.DataFrame(rows, columns=column)
        return df

    def generate_df_avg_word_length(self, dataset, exponent=1):
        """ Generates a 10000x1 df for the average word length

        Args:
            dataset: list of dicts containing reddit comments and
            their attributes

        Returns:
            df: 10000x1 pandas df that contains the word length
            for every training instance
        """
        if exponent == 1:
            column = ['word_length']
        elif exponent > 1:
            column = ['word_length**' + str(exponent)]

        rows = []
        for data_point in dataset:
            for info_name, info_value in data_point.items():
                if info_name == 'text':
                    info_value = self.array_to_str(info_value)
                    words = info_value.split()
                    word_length = sum(len(word)for word in words) / len(words)
                    if exponent > 1:
                        word_length = word_length ** exponent
            rows.append(word_length)

        df = pd.DataFrame(rows, columns=column)
        return df

    def generate_df_nb_words_per_sent(self, dataset, exponent=1):
        """ Generates a 10000x1 df for the avg number of words per sentence

        Args:
            dataset: list of dicts containing reddit comments and
            their attributes

        Returns:
            df: 10000x1 pandas df that contains the averaged number of
            words per sentence for every training instance
        """
        if exponent == 1:
            column = ['nb_word_per_sent']
        elif exponent > 1:
            column = ['nb_word_per_sent**' + str(exponent)]

        rows = []
        for data_point in dataset:
            for info_name, info_value in data_point.items():
                if info_name == 'text':
                    info_value = self.array_to_str(info_value)
                    sents = info_value.replace(
                        '?', '.').replace('!', '.').split('.')
                    sents = list(filter(None, sents))
                    if len(sents) > 0:
                        nb_words_per_sent = sum(
                            len(x.split())for x in sents) / len(sents)
                    else:
                        nb_words_per_sent = 1
                    if exponent > 1:
                        nb_words_per_sent = nb_words_per_sent ** exponent
            rows.append(nb_words_per_sent)

        df = pd.DataFrame(rows, columns=column)
        return df

    def generate_df_sentiment(self, dataset, exponent=1):
        """ Generates a 10000x1 df for the sentiment_score

        Args:
            dataset: list of dicts containing reddit comments and
            their attributes

        Returns:
            df: 10000x1 pandas df that contains sentiment_score
            for every training instance
        """
        if exponent == 1:
            column = ['sentiment']
        elif exponent > 1:
            column = ['sentiment**' + str(exponent)]

        rows = []
        for data_point in dataset:
            for info_name, info_value in data_point.items():
                if info_name == 'text':
                    info_value = self.array_to_str(info_value)
                    sentiment_score = TextBlob(info_value).sentiment.polarity
                    if exponent > 1:
                        sentiment_score = sentiment_score ** exponent
            rows.append(sentiment_score)

        df = pd.DataFrame(rows, columns=column)
        return df

    def generate_df_readability(self, dataset, exponent=1):
        """ Generates a 10000x1 df for the readability score (Gunning Fox index)

        Args:
            dataset: list of dicts containing reddit comments and
            their attributes

        Returns:
            df: 10000x1 pandas df that contains readability_score
            for every training instance
        """
        if exponent == 1:
            column = ['readability']
        elif exponent > 1:
            column = ['readability**' + str(exponent)]

        rows = []
        for data_point in dataset:
            for info_name, info_value in data_point.items():
                if info_name == 'text':
                    info_value = self.array_to_str(info_value)
                    readability_score = textstat.gunning_fog(info_value)
                    if exponent > 1:
                        readability_score = readability_score ** exponent
            rows.append(readability_score)

        df = pd.DataFrame(rows, columns=column)
        return df

    def generate_interact(self, dataset1, col1, dataset2, col2):
        """ Generates a 10000x1 df for the product of col1 and string2

        Args:
            dataset: list of dicts containing exist features

        Raises:
            InputError: If the input colume name col1 or col2 does not exist.

        Returns:
            df: 10000x1 pandas df that contains the product of col1 and col2
            for every training instance
        """
        column = [col1+'*'+col2]
        if col1 not in dataset1:
            raise ValueError('1st argument does not exist in dataset1.')
        if col2 not in dataset2:
            raise ValueError('2nd argument does not exist in dataset2.')
        df = pd.DataFrame(
            (dataset1[col1]*dataset2[col2]).values, columns=column)
        return df

    # extra features ends here

    def generate_df_most_frequent_words(self, dataset, most_frequent_words):
        """ Generates a 10000xN df for the most frequent words

        The df contains one row for every training instance and one
        columnfor each word in the top N. The result is a 10000xN matrix
        that contains occurences of the most common words for every reddit
        comment

        Args:
            dataset: list of dicts containing reddit comments and
            their attributes
            most_frequent_words: list of tuples containing the most
            frequent words and their occurences

        Returns:
            df: 10000xN pandas df that contains occurences of the
            most frequent words for every training instance
        """
        columns = []
        for item in most_frequent_words:
            columns.append(item)

        rows = []
        for data_point in dataset:
            for info_name, info_value in data_point.items():
                if info_name == 'text':
                    occurences = []
                    for item in most_frequent_words:
                        occurences.append(np.count_nonzero(info_value == item))
            rows.append(occurences)

        df = pd.DataFrame(rows, columns=columns)
        return df

    def generate_df_bias(self, dataset):
        """ Generates a 10000x1 df for the number of children comments

        Args:
            dataset: list of dicts containing reddit comments and their
            attributes
        Returns:
            df: 10000x1 pandas df that contains only ones and serves
            as the bias term
        """

        ones_data = np.ones(len(dataset))
        df = pd.DataFrame(ones_data, columns=['bias'])
        return df

    def generate_df_children(self, dataset, exponent=1):
        """ Generates a 10000x1 df for the number of children comments

        Args:
            dataset: list of dicts containing reddit comments and
            their attributes

        Returns:
            df: 10000x1 pandas df that contains the number of children
            comments for every training instance
        """
        if exponent == 1:
            column = ['children']
        elif exponent > 1:
            column = ['children**' + str(exponent)]

        rows = []
        for data_point in dataset:
            for info_name, info_value in data_point.items():
                if info_name == 'children':
                    children = info_value
                    if exponent > 1:
                        children = children ** exponent
            rows.append(children)

        df = pd.DataFrame(rows, columns=column)
        return df

    def generate_df_controversiality(self, dataset, exponent=1):
        """ Generates a 10000x1 df for the number of controversiality

        Args:
            dataset: list of dicts containing reddit comments and
            their attributes

        Returns:
            df: 10000x1 pandas df that contains the controversiality
            value for every training instance
        """
        if exponent == 1:
            column = ['controversiality']
        elif exponent > 1:
            column = ['controversiality**' + str(exponent)]

        rows = []
        for data_point in dataset:
            for info_name, info_value in data_point.items():
                if info_name == 'controversiality':
                    controversiality = info_value
                    if exponent > 1:
                        controversiality = controversiality ** exponent
            rows.append(controversiality)

        df = pd.DataFrame(rows, columns=column)
        return df

    def generate_df_is_root(self, dataset, exponent=1):
        """ Generates a 10000x1 df for the feature is_root

        Args:
            dataset: list of dicts containing reddit comments and
            their attributes

        Returns:
            df: 10000x1 pandas df that contains is_root
            for every training instance
        """
        if exponent == 1:
            column = ['is_root']
        elif exponent > 1:
            column = ['is_root**' + str(exponent)]

        rows = []
        for data_point in dataset:
            for info_name, info_value in data_point.items():
                if info_name == 'is_root':
                    if info_value is False:
                        is_root = 0
                    elif info_value is True:
                        is_root = 1
            rows.append(is_root)

        df = pd.DataFrame(rows, columns=column)
        return df

    def generate_df_popularity_score(self, dataset):
        """ Generates a 10000x1 df for the popularity_score

        Args:
            dataset: list of dicts containing reddit comments and
            their attributes

        Returns:
            df: 10000x1 pandas df that contains popularity_score
            for every training instance
        """
        column = ['popularity_score']

        rows = []
        for data_point in dataset:
            for info_name, info_value in data_point.items():
                if info_name == 'popularity_score':
                    popularity_score = info_value
            rows.append(popularity_score)

        df = pd.DataFrame(rows, columns=column)
        return df

    def get_160_most_frequent_words(self, dataset):
        """ Gets a list containing the 160 most frequent words and their occurences

        Process reddit comments by applying lower() and split() on them, then
        returns the most frequent words and their occurences from the dataset

        Args:
            dataset: list of dicts containing reddit comments and
            their attributes

        Returns:
            most_frequent_words: list of tuples containing the most
            frequent words and their occurences
        """
        all_text = np.concatenate([item['text'] for item in dataset])
        unique_text, count = np.unique(all_text, return_counts=True)
        most_frequent_words = unique_text[np.argsort(-count)][:160:1]
        most_frequent_words = np.array(most_frequent_words)
        return most_frequent_words.tolist()

    def flatten_list_of_lists(l):
        return [item for sublist in l for item in sublist]

    def data_point_to_str(self, data_point):
        """ Prints info about a data point

        A data point is a dictionary with the following attributes:
        popularity_score : float representiing the popualrity score
        children : the number of replies to this comment (type: int)
        text : the text of this comment (type: string)
        controversiality : '0' or '1' representing wether the comment
        is controversial
        is_root : 'true' if comment is a reply to a post, 'false' if it's
        a reply to a thread

        Args:
            data_point: dict containing a data point from the dataset

        Returns:
            s: string containing the available info about the data point
        """
        s = ''
        for info_name, info_value in data_point.items():
            s += info_name + ' : ' + str(info_value) + '\n'
        return s

    def create_wordstxt(self):
        """ Creates the word.txt file of 160 most frequent words in training set
        """
        with open('words.txt', 'w', encoding="utf-8") as fp:
            for word in self.top_160_words:
                fp.write(word + '\n')

    def array_to_str(self, array):
        string = ""
        for item in array:
            string += item + ' '
        return string[:-1]

    def process_text(self, string):
        """ Lowers() and splits() a string """
        processed_text = np.array(string.lower().split())
        return processed_text

    def get_training_dataset(self):
        """ Returns partitioned training dataset"""
        return self.dataset[:10000]

    def get_validation_dataset(self):
        """ Returns partitioned validation dataset"""
        return self.dataset[10000:11000]

    def get_test_dataset(self):
        """ Returns partitioned testing dataset"""
        return self.dataset[11000:]
