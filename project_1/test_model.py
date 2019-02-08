import os
import pandas as pd
import numpy as np
import implement_models
import process_dataset

input_data_path = os.path.join(os.path.curdir, 'data', 'proj1_data.json')
data_processor = process_dataset.DataProcessor(input_data_path)

training_dataset = data_processor.get_training_dataset()
validation_dataset = data_processor.get_validation_dataset()

word_count = data_processor.generate_df_word_count(training_dataset)
nb_words_per_sent = data_processor.generate_df_nb_words_per_sent(
    training_dataset)
word_length = data_processor.generate_df_avg_word_length(training_dataset)
children = data_processor.generate_df_children(training_dataset)
controversiality = data_processor.generate_df_controversiality(
    training_dataset)
sentiment = data_processor.generate_df_sentiment(training_dataset)
readability = data_processor.generate_df_readability(training_dataset)
sentiment_2 = data_processor.generate_interact(
    sentiment, 'sentiment', sentiment, 'sentiment')
children_sentiment = data_processor.generate_interact(
    children, 'children', sentiment, 'sentiment')
bias = data_processor.generate_df_bias(training_dataset)
features_df = pd.concat(
    [controversiality, sentiment, sentiment_2, children_sentiment, bias],
    axis=1)


output_df = data_processor.generate_df_output(training_dataset)
modeling = implement_models.Modeling()

model = modeling.generate_linear_model(
    features_df, output_df)
print('ref: ', model.coef_, model.intercept_)
print('R2 score: ', model.score(features_df, output_df))


predict = modeling.generate_closed_form_regression(features_df, output_df)
print('closed form: ', predict)

ones_data = np.ones(len(features_df.columns))
w_init_df = pd.DataFrame(predict.values+0.05, columns=['w_init'])
(predict, err_history) = modeling.generate_gradient_descent_regression(
    features_df, output_df, w_init_df, 0.2)
print('gradient: ', predict)
print(err_history)
