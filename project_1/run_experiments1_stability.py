# %% Notebook
import os
import pandas as pd
import process_dataset
import implement_models
import time
import numpy as np
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
Experiment 1: Stability study for the gradient descent method
"""
print("----------------------")
print("Experiment 1-Stability")
print("----------------------")
# First, we study a model that contains no text features

training_features, training_output = dp.generate_df_features(
    training_dataset, top_words=0, features=[0,0,0,0,0,0,0,0,0])
validation_features, validation_output = dp.generate_df_features(
    validation_dataset, top_words=0, features=[0,0,0,0,0,0,0,0,0])


start = time.time()
w_ref=modeling.generate_closed_form_regression(training_features, training_output)
end = time.time()
elapsed = end - start
ref_training_MSE, training_R2 = modeling.error_estimator(
                     training_features, w_ref, training_output)
validation_MSE, validation_R2 = modeling.error_estimator(
             validation_features, w_ref, validation_output)
print('ref_MSE: '+str(ref_training_MSE))
#print(modeling.error_estimator(validation_features,w,validation_output))
test_number=8
table_iter=np.eye(test_number)
table_err=np.eye(test_number)
for i in range(test_number):
    dist=(i+1)*0.3
    for j in range(test_number):
        rate=(j+1)*0.1
        w_init = pd.DataFrame(w_ref.values+dist, columns=['w_init'])
        w, MSE_history = modeling.generate_gradient_descent_regression(
                training_features, training_output, w_init, rate, 0)
        training_MSE, training_R2 = modeling.error_estimator(
                training_features, w, training_output)
        validation_MSE, validation_R2 = modeling.error_estimator(
                validation_features, w, validation_output)
        table_iter[i,j]=len(MSE_history.values)
        table_err[i,j]=round(training_MSE-ref_training_MSE,3)
print('Nb of iterations for fixed learning rate')
print(table_iter)
print('MSE error from reference for fixed learning rate')
print(table_err)
for i in range(test_number):
    dist=(i+1)*0.3
    for j in range(test_number):
        rate=(j+1)*0.1
        w_init = pd.DataFrame(w_ref.values+dist, columns=['w_init'])
        w, MSE_history = modeling.generate_gradient_descent_regression(
                training_features, training_output, w_init, rate, 1)
        training_MSE, training_R2 = modeling.error_estimator(
                training_features, w, training_output)
        validation_MSE, validation_R2 = modeling.error_estimator(
                validation_features, w, validation_output)
        table_iter[i,j]=len(MSE_history.values)
        table_err[i,j]=round(training_MSE-ref_training_MSE,3)
print('Nb of iterations for adaptive learning rate')
print(table_iter)
print('MSE error from reference for adaptive learning rate')
print(table_err)
