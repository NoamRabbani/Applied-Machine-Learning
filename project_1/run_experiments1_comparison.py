# %% Notebook
import os
import pandas as pd
import process_dataset
import implement_models
import time
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
        print('no display found. Using non-interactive Agg backend')
        mpl.use('Agg')
import matplotlib.pyplot as plt

# Load the dataset and partition it into training, valdiation and testing
input_data_path = os.path.join(os.path.curdir, 'data', 'proj1_data.json')
dp = process_dataset.DataProcessor(input_data_path)
modeling = implement_models.Modeling()

training_dataset = dp.get_training_dataset()
validation_dataset = dp.get_validation_dataset()
test_dataset = dp.get_test_dataset()

# dp.create_wordstxt()


"""
Experiment 1: Compare the runtime and performance of the implemented regression models
"""
print("-----------------------")
print("Experiment 1-Comparison")
print("-----------------------")

training_features, training_output = dp.generate_df_features(
    training_dataset, top_words=0, features=[0,0,0,0,0,0,0,0,0])
validation_features, validation_output = dp.generate_df_features(
    validation_dataset, top_words=0, features=[0,0,0,0,0,0,0,0,0])

start = time.time()
ref_model = modeling.generate_linear_model(
    training_features, training_output)
end = time.time()
elapsed = end - start
training_MSE=mean_squared_error(training_output,ref_model.predict(training_features))
validation_MSE=mean_squared_error(validation_output,ref_model.predict(validation_features))
validation_R2=r2_score(validation_output,ref_model.predict(validation_features))
print("*************** ")
print("Reference****** ")
print("*************** ")
print("Time: " + str(elapsed))
print("**On training set: ")
print("MSE: " + str(training_MSE))
#print("R2: " +
#      str(ref_model.score(training_features, training_output)))
print("**On validation set: ")
print("MSE: " + str(validation_MSE))
#print("R2: " + str(validation_R2))

start = time.time()
w=modeling.generate_closed_form_regression(training_features, training_output)
end = time.time()
elapsed = end - start
training_MSE, training_R2 = modeling.error_estimator(
    training_features, w, training_output)
validation_MSE, validation_R2 = modeling.error_estimator(
    validation_features, w, validation_output)
print("***************** ")
print("Closed form****** ")
print("*****************")
print("Time: " + str(elapsed))
print("**On training set: ")
print("MSE: " + str(training_MSE))
#print("R2: " + str(training_R2))
print("**On validation set: ")
print("MSE: " + str(validation_MSE))
#print("R2: " + str(validation_R2))


w_init = pd.DataFrame(w.values+0.3, columns=['w_init'])
start = time.time()
w, MSE_history = modeling.generate_gradient_descent_regression(
    training_features, training_output, w_init, 1.,1)
end = time.time()
elapsed = end - start
training_MSE, training_R2 = modeling.error_estimator(
    training_features, w, training_output)
validation_MSE, validation_R2 = modeling.error_estimator(
    validation_features, w, validation_output)
print("********************** ")
print("Gradient descent****** ")
print("********************** ")
print("Time: " + str(elapsed))
print("**On training set: ")
print("MSE: " + str(training_MSE))
#print("R2: " + str(training_R2))
print("**On validation set: ")
print("MSE: " + str(validation_MSE))
#print("R2: " + str(validation_R2))
plt.plot(MSE_history.values)
plt.xlabel('Iter')
plt.ylabel('MES')
plt.grid(True)
plt.savefig('grad_desc_conv.png')

plt.loglog(MSE_history.values)
plt.xlabel('Iter')
plt.ylabel('MES')
plt.grid(True)
plt.savefig('grad_desc_conv_log.png')
