from sklearn import linear_model
import numpy as np
import pandas as pd
import process_dataset


class Modeling():
    """ Handles the modeling operations on the features matrix X
    """

    def generate_linear_model(self, features_df, output_df):
        """Generates a linear regression model using sklearn library

        Args:
           features_df: the dataframe containing features
           output_df: the dataframe containing the output

        Returns:
            model: sklean linear regression model
        """
        model = linear_model.LinearRegression().fit(features_df, output_df)
        return model

    def generate_closed_form_regression(self, features_df, output_df):
        """Generates a linear regression model using the closed-form

        Args:
           features_df: the dataframe containing features
           output_df: the dataframe containing the output

        Returns:
            df: the dataframe contains the parameters/weights
            MSE: the dataframe contains the Mean Squared Error
            R2: the dataframe contains the final R2 score
        """
        X = features_df.values
        Y = output_df.values

        column = ['weights']
        column_mse = ['MSE']
        column_r2 = ['R2']
        w = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)
        MSE = pd.DataFrame(
            (Y-X.dot(w)).transpose().dot(Y-X.dot(w))[0], columns=column_mse)
        R2 = pd.DataFrame(
            (w.transpose().dot(X.transpose()).dot(Y)-(Y.sum())**2/len(Y)) /
            (Y.transpose().dot(Y)-(Y.sum())**2/len(Y)), columns=column_r2)
        df = pd.DataFrame(w, columns=column)
        return df  # , MSE, R2

    def generate_gradient_descent_regression(
            self, features_df, output_df, w_init, learning_rate, vary):
        """Generates a linear regression model using the gradient descent

        Args:
           features_df: the dataframe containing features
           output_df: the dataframe containing the output
           w_init: the initial guess of the weights
           learning_rate: learning rate for gradient descent

        Returns:
            df_w: the dataframe contains the parameters/weight
            df_err: the dataframe contains the error at every iteration
            R2: the dataframe contains the final R2 score
        """
        X = features_df.values
        Y = output_df.values

        w = w_init.values
        column_w = ['weights']
        column_err = ['MSE']
        column_r2 = ['R2']
        iter = 1
        epsi = 1E-6
        err = np.ones(len(w_init))
        err_old = np.zeros(len(w_init))
        err_history = []
        while abs(np.linalg.norm(err-err_old)) > epsi and iter < 5000:
            # rescale the learning rate with the length of outputs
            if vary==0:
                alpha = learning_rate
            else:
                alpha = learning_rate*1./(iter+1.)
            err_old = err
            err = (Y-X.dot(w)).transpose().dot(Y-X.dot(w))[0]/len(Y)
            err_grad = 2*(X.transpose().dot(X).dot(w)-X.transpose().dot(Y))/len(Y)
            w = w-alpha*err_grad
            iter += 1
            err_history.append(err)
        R2 = pd.DataFrame(
            (w.transpose().dot(X.transpose()).dot(Y)-(Y.sum())**2/len(Y)) /
            (Y.transpose().dot(Y)-(Y.sum())**2/len(Y)), columns=column_r2)
        df_w = pd.DataFrame(w, columns=column_w)
        df_err = pd.DataFrame(err_history, columns=column_err)
        return df_w, df_err#, R2

    def error_estimator(self, features_df, w_df, output_df):
        """Generates the error estimation for the given model applied to
           other dataset

        Args:
           features_df: the dataframe containing features
           output_df: the dataframe containing the output
           w: the weights of the given model

        Returns:
            df_w: the dataframe contains the parameters/weight
            df_err: the dataframe contains the error at every iteration
            R2: the dataframe contains the final R2 score
        """
        X = features_df.values
        Y = output_df.values
        w = w_df.values

        column_mse = ['MSE']
        column_r2 = ['R2']

        MSE = pd.DataFrame(
            (Y-X.dot(w)).transpose().dot(Y-X.dot(w))[0], columns=column_mse)
        R2 = pd.DataFrame(
            (w.transpose().dot(X.transpose()).dot(Y)-(Y.sum())**2/len(Y)) /
            (Y.transpose().dot(Y)-(Y.sum())**2/len(Y)), columns=column_r2)
        return (MSE['MSE'].iloc[0])/len(Y), R2['R2'].iloc[0]
