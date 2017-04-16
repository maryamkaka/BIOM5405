import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from sklearn.metrics import precision_recall_curve
from scipy.interpolate import UnivariateSpline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

OUTPUT_LOCATION = '/images/'


def q2c(n=1000, plot_pdf=False, prev_corrected=False):
    """"""
    pcPR = lambda sn, r, sp: sn/(sn + r*(1-sp))
    pos = np.random.normal(50, 5, n)
    neg = np.random.normal(60, 12, n)

    if plot_pdf:
        pos = stats.norm(loc=50, scale=5)
        neg = stats.norm(loc=60, scale=12)
        x = np.linspace(0, 100, 1000)
        plt.figure()
        plt.title('PDF of Class Distributions')
        plt.ylabel('Probability')
        plt.grid(True)
        plt.plot(x, pos.pdf(x))
        plt.plot(x, neg.pdf(x))
        plt.savefig('2ci.png', bbox_inches='tight')


def q3(plots=False):
    """
    Assume that you have measured three features for 100 samples each of two 
    different types of fish. The data for class 1 (salmon) is found in 
    final_Q3_data1.tsv and the data for class 2 (trout) is found in 
    final_Q3_data2.tsv. Rows are samples and columns are features.
    
    You decide to use a Bayesian classifier with the assumption that the two 
    class-conditional distributions follow multivariate normal distributions 
    with equal priors. Estimate the mean vector and covariance matrix for each
    class-conditional distribution. Compute the determinant and inverse of the 
    covariance matrices. What is wrong How can you fix it? (hint, visualize 
    your data). Discuss what was wrong, how you fixed it, and give the apparent
    error rate for both your original and ‘fixed’ classifiers. (300 words, 
    include a plot illustrating the “problem” with the feature data).
    
    :param:
    plots: binary, default:False
    
    :return: 
    """
    salmon = pd.read_csv("final_Q3_data1.tsv", sep='\t', index_col=False,
                         header=None)
    trout = pd.read_csv("final_Q3_data2.tsv", sep='\t', index_col=False,
                        header=None)

    # visualize data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(salmon[0], salmon[1], salmon[2], label='Salmon')
    ax.scatter(trout[0], trout[1], trout[2], c='b', marker='^',
               label='Trout')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.legend(loc='upper left')
    if plots: plt.show()

    plt.figure()
    plt.scatter(salmon[0], salmon[1], label='Salmon')
    plt.scatter(trout[0], trout[1], c='b', marker='^', label='Trout')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='lower right')
    plt.title('Plot of Data')
    plt.grid(True)
    if plots: plt.show()

    # set labels (Positive Class: Salmon; Negative Class: Trout)
    salmon[[salmon.shape[1]-1]] = 1
    trout[[trout.shape[1]-1]] = 0

    # finding means and cov

    # train classifier
    x_train, x_test, y_train, y_test = train_test_split(
        pd.concat([salmon[[0, 1, 2]], trout[[0,1,2]]]),
        pd.concat([salmon[[3]], trout[[3]]]),
        train_size=0.7,
        random_state=42
    )

    c = GaussianNB()
    c.fit(x_train, y_train)
    print('Baysian Classifier - Whitening Transform - Accuracy: ')
    print(c.score(x_test, y_test))

    # fixed classifier
    salmon_t = PCA().fit_transform(salmon[[0,1,2]])
    trout_t = PCA().fit_transform(trout[[0, 1, 2]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(salmon_t[:, 0], salmon_t[:,1], salmon_t[:,2], label='Salmon')
    ax.scatter(trout_t[:, 0], trout_t[:, 1], trout_t[:, 2], c='b', marker='^',
               label='Trout')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.legend(loc='upper left')
    if plots: plt.show()

    x_train, x_test, y_train, y_test = train_test_split(
        np.concatenate((salmon_t, trout_t)),
        pd.concat([salmon[[3]], trout[[3]]]),
        train_size=0.7,
        random_state=42
    )

    c = GaussianNB()
    c.fit(x_train, y_train)
    print('Fixed Baysian Classifier Accuracy: ')
    print(c.score(x_test, y_test))


def main():
    """ """
    print('  QUESTION 2:  ')
    # q2c(plot_pdf=True)

    print('  QUESTION 3:  ')
    q3(plots=True)


if __name__ == "__main__":
    main()