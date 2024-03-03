import random as rnd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

class SimpleBayesClassifier:

    def __init__(self, n_pos, n_neg):
        
        """
        Initializes the SimpleBayesClassifier with prior probabilities.

        Parameters:
        n_pos (int): The number of positive samples.
        n_neg (int): The number of negative samples.
        
        Returns:
        None: This method does not return anything as it is a constructor.
        """

        self.n_pos = n_pos
        self.n_neg = n_neg
        self.prior_pos = n_pos / (n_pos + n_neg)
        self.prior_neg = n_neg / (n_pos + n_neg)

    def fit_params(self, x, y, n_bins = 10, alpha=1e-6):

        """
        Computes histogram-based parameters for each feature in the dataset.

        Parameters:
        x (np.ndarray): The feature matrix, where rows are samples and columns are features.
        y (np.ndarray): The target array, where each element corresponds to the label of a sample.
        n_bins (int): Number of bins to use for histogram calculation.

        Returns:
        (stay_params, leave_params): A tuple containing two lists of tuples, 
        one for 'stay' parameters and one for 'leave' parameters.
        Each tuple in the list contains the bins and edges of the histogram for a feature.
        """
        def calculate_histogram_params(x, n_bins, is_stay=True, alpha=1e-6):
            x = x[~np.isnan(x)]
            bin_edges = np.linspace(x.min(), x.max(), n_bins-1)
            # make sure your first bin should cover -inf and the last bin should cover inf
            bin_edges = np.insert(bin_edges, 0, -np.inf)
            bin_edges = np.insert(bin_edges, len(bin_edges), np.inf)
            bin_idx = np.digitize(x, bin_edges)
            
            # count the number of data points in each bin
            hist = np.bincount(bin_idx)[1:]

            # calculate the probability of each bin
            prior = self.prior_pos if is_stay else self.prior_neg
            hist = (hist + (alpha * prior)) / (hist.sum() + (2 * alpha))
            return hist, bin_edges

        self.stay_params = [(None, None) for _ in range(x.shape[1])]
        self.leave_params = [(None, None) for _ in range(x.shape[1])]

        x_stay = x[y == 0]
        x_leave = x[y == 1]

        for i in range(x.shape[1]):
            self.stay_params[i] = calculate_histogram_params(x_stay[:, i], n_bins, True, alpha)
            self.leave_params[i] = calculate_histogram_params(x_leave[:, i], n_bins, False, alpha)

        return self.stay_params, self.leave_params

    def predict(self, x, thresh = 0):

        """
        Predicts the class labels for the given samples using the non-parametric model.

        Parameters:
        x (np.ndarray): The feature matrix for which predictions are to be made.
        thresh (float): The threshold for log probability to decide between classes.

        Returns:
        result (list): A list of predicted class labels (0 or 1) for each sample in the feature matrix.
        """

        y_pred = []
        for i in range(x.shape[0]):
            log_prob_stay = np.log(self.prior_neg)
            log_prob_leave = np.log(self.prior_pos)
            for j in range(x.shape[1]):
                if np.isnan(x[i, j]):
                    continue
                hist_stay, bin_edges_stay = self.stay_params[j]
                hist_leave, bin_edges_leave = self.leave_params[j]

                bin_idx_stay = np.digitize(x[i, j], bin_edges_stay) - 1
                bin_idx_leave = np.digitize(x[i, j], bin_edges_leave) - 1
                
                log_prob_stay += np.log(hist_stay[bin_idx_stay])
                log_prob_leave += np.log(hist_leave[bin_idx_leave])
                
            y_pred.append(1 if log_prob_leave - log_prob_stay > thresh else 0)
        return np.array(y_pred)
    
    def fit_gaussian_params(self, x, y):

        """
        Computes mean and standard deviation for each feature in the dataset.

        Parameters:
        x (np.ndarray): The feature matrix, where rows are samples and columns are features.
        y (np.ndarray): The target array, where each element corresponds to the label of a sample.

        Returns:
        (gaussian_stay_params, gaussian_leave_params): A tuple containing two lists of tuples,
        one for 'stay' parameters and one for 'leave' parameters.
        Each tuple in the list contains the mean and standard deviation for a feature.
        """

        self.gaussian_stay_params = [(0, 0) for _ in range(x.shape[1])]
        self.gaussian_leave_params = [(0, 0) for _ in range(x.shape[1])]

        x_stay = x[y == 0]
        x_leave = x[y == 1]

        for i in range(x.shape[1]):
            self.gaussian_stay_params[i] = (np.nanmean(x_stay[:, i]), np.nanstd(x_stay[:, i]))
            self.gaussian_leave_params[i] = (np.nanmean(x_leave[:, i]), np.nanstd(x_leave[:, i]))
        
        return self.gaussian_stay_params, self.gaussian_leave_params
    
    def gaussian_predict(self, x, thresh = 0):

        """
        Predicts the class labels for the given samples using the parametric model.

        Parameters:
        x (np.ndarray): The feature matrix for which predictions are to be made.
        thresh (float): The threshold for log probability to decide between classes.

        Returns:
        result (list): A list of predicted class labels (0 or 1) for each sample in the feature matrix.
        """

        y_pred = []

        for i in range(x.shape[0]):
            log_prob_stay = np.log(self.prior_neg)
            log_prob_leave = np.log(self.prior_pos)
            for j in range(x.shape[1]):
                if np.isnan(x[i, j]):
                    continue
                mean_stay, std_stay = self.gaussian_stay_params[j]
                mean_leave, std_leave = self.gaussian_leave_params[j]
                log_prob_stay += np.log(stats.norm(mean_stay, std_stay).pdf(x[i, j]))
                log_prob_leave += np.log(stats.norm(mean_leave, std_leave).pdf(x[i, j]))
            y_pred.append(1 if log_prob_leave - log_prob_stay > thresh else 0)

        return np.array(y_pred)