# %% Notebook
import os
import pandas as pd
import process_dataset
import implement_models
from sklearn.metrics import mean_squared_error

# Load the dataset and partition it into training, valdiation and testing
input_data_path = os.path.join(os.path.curdir, 'data', 'proj1_data.json')
dp = process_dataset.DataProcessor(input_data_path)
modeling = implement_models.Modeling()

training_dataset = dp.get_training_dataset()
validation_dataset = dp.get_validation_dataset()
test_dataset = dp.get_test_dataset()

# dp.create_wordstxt()

"""
Experiment 2: Performance of the closed-form model based on the number of
features
"""

print("---------------")
print("Experiment 2")
print("---------------")
for top_words in [0, 60, 160]:
    train_features, train_output = dp.generate_df_features(
        training_dataset, top_words)
    val_features, val_output = dp.generate_df_features(
        validation_dataset, top_words)

    weights = modeling.generate_closed_form_regression(
        train_features, train_output)

    training_MSE, training_R2 = modeling.error_estimator(
        train_features, weights, train_output)
    validation_MSE, validation_R2 = modeling.error_estimator(
        val_features, weights, val_output)
    print("Training & Validation MSE for top_words=" + str(top_words) +
          " : " + str(training_MSE) + " & " + str(validation_MSE))

# According to our results, the best performing model is the closed form
# approach using the basic features in addition to the top 60 words

"""
Experiment 3: Closed-form model improvement from the addition of new features
A new feature needs to reduce the MSE below 0.9789 to reach the 0.005
improvement treshold
"""
print("---------------")
print("Experiment 3")
print("---------------")

features = [[2, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 0, 0, 0, 2, 0],
            [2, 0, 0, 0, 0, 1, 0, 0, 0], [2, 0, 0, 0, 0, 1, 0, 0, 1]]

for combination in features:
    train_features, train_output = dp.generate_df_features(
        training_dataset, top_words=60, features=combination)
    val_features, val_output = dp.generate_df_features(
        validation_dataset, top_words=60, features=combination)

    weights = modeling.generate_closed_form_regression(
        train_features, train_output)

    training_MSE, training_R2 = modeling.error_estimator(
        train_features, weights, train_output)
    validation_MSE, validation_R2 = modeling.error_estimator(
        val_features, weights, val_output)
    print("Training & Validation MSE for extra feature combination " +
          str(combination) + ": " + str(training_MSE) + " & " +
          str(validation_MSE))

"""
Experiment 3: Running the best model on the test set
"""
print("---------------")
print("Experiment 4")
print("---------------")

features = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [2, 0, 0, 0, 0, 1, 0, 0, 1]]

for combination in features:
    train_features, train_output = dp.generate_df_features(
        training_dataset, top_words=60, features=combination)
    val_features, val_output = dp.generate_df_features(
        validation_dataset, top_words=60, features=combination)
    test_features, test_output = dp.generate_df_features(
        test_dataset, top_words=60, features=combination)

    weights = modeling.generate_closed_form_regression(
        train_features, train_output)

    training_MSE, training_R2 = modeling.error_estimator(
        train_features, weights, train_output)
    validation_MSE, validation_R2 = modeling.error_estimator(
        val_features, weights, val_output)
    test_MSE, test_R2 = modeling.error_estimator(
        test_features, weights, test_output)
    print("Training & Validation & Test MSE for extra feature combination " +
          str(combination) + ": " + str(training_MSE) + " & " +
          str(validation_MSE) + " & " + str(test_MSE))
