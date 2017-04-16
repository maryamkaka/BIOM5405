import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from sklearn.metrics import precision_recall_curve, mean_squared_error
from scipy.interpolate import UnivariateSpline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

OUTPUT_LOCATION = '/images/'


def q2c(n=1000, plot_pdf=False, prev_corrected=False):
    """"""
    pcPR = lambda sn, r, sp: sn/(sn + r*(1-sp))
    pos = np.random.normal(50, 5, n)
    neg = np.random.normal(60, 12, n)

    pos = stats.norm(loc=50, scale=5)
    neg = stats.norm(loc=60, scale=12)
    x = np.linspace(0, 100, 1000)
    plt.figure()
    plt.title('PDF of Class Distributions')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.plot(x, pos.pdf(x))
    plt.plot(x, neg.pdf(x))
    if plot_pdf:
        plt.show()
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
    plt.subplot(1, 3, 1)
    plt.scatter(salmon[0], salmon[1], label='Salmon')
    plt.scatter(trout[0], trout[1], c='b', marker='^', label='Trout')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.title('Plot of Data')
    plt.scatter(salmon[0], salmon[2], label='Salmon')
    plt.scatter(trout[0], trout[2], c='b', marker='^', label='Trout')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 3')
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.scatter(salmon[1], salmon[2], label='Salmon')
    plt.scatter(trout[1], trout[2], c='b', marker='^', label='Trout')
    plt.xlabel('Feature 2')
    plt.ylabel('Feature 3')
    plt.grid(True)
    plt.legend(loc='lower right')
    if plots: plt.show()

    # set labels (Positive Class: Salmon; Negative Class: Trout)
    salmon[[salmon.shape[1]-1]] = 1
    trout[[trout.shape[1]-1]] = 0

    # finding means and cov

    # train classifiers
    x = pd.concat([salmon[[0, 1, 2]], trout[[0, 1, 2]]])
    y = pd.concat([salmon[[3]], trout[[3]]])
    c = GaussianNB()
    c.fit(x, y)
    print('1 - Bayesian Classifier - Error: ' +
          str(mean_squared_error(y, c.predict(x))))

    x = pd.concat([salmon[[1, 2]], trout[[1, 2]]])
    y = pd.concat([salmon[[3]], trout[[3]]])
    c = GaussianNB()
    c.fit(x, y)
    print('2 - Bayesian Classifier + Remove F1 - Error: ' +
          str(mean_squared_error(y, c.predict(x))))

    x = pd.concat([salmon[[0, 2]], trout[[0, 2]]])
    y = pd.concat([salmon[[3]], trout[[3]]])
    c = GaussianNB()
    c.fit(x, y)
    print('3 - Bayesian Classifier + Remove F2 - Error: ' +
          str(mean_squared_error(y, c.predict(x))))

    x = pd.concat([salmon[[0, 1]], trout[[0, 1]]])
    y = pd.concat([salmon[[3]], trout[[3]]])
    c = GaussianNB()
    c.fit(x, y)
    print('4 - Bayesian Classifier + Remove F3 - Error: ' +
          str(mean_squared_error(y, c.predict(x))))

    # Normalization
    x = np.concatenate([normalize(salmon[[0, 1, 2]]),
                        normalize(trout[[0, 1, 2]])])
    y = pd.concat([salmon[[3]], trout[[3]]])
    c = GaussianNB()
    c.fit(x, y)
    print('5 - Bayesian Classifier + Normalization - Error: ' +
          str(mean_squared_error(y, c.predict(x))))

    # Normalization
    x = np.concatenate([normalize(salmon[[2, 1]]),
                        normalize(trout[[2, 1]])])
    y = pd.concat([salmon[[3]], trout[[3]]])
    c = GaussianNB()
    c.fit(x, y)
    print('6 - Bayesian Classifier + Normalization + Remove F1 - Error: ' +
          str(mean_squared_error(y, c.predict(x))))

    x = np.concatenate([normalize(salmon[[0, 1]]),
                        normalize(trout[[0, 1]])])
    y = pd.concat([salmon[[3]], trout[[3]]])
    c = GaussianNB()
    c.fit(x, y)
    print('7 - Bayesian Classifier + Normalization + Remove F3 - Error: ' +
          str(mean_squared_error(y, c.predict(x))))

    # Bayesian Classifier + Whitening Transform
    salmon_t = PCA().fit_transform(salmon[[0, 1, 2]])
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

    x = np.concatenate((salmon_t, trout_t))
    y = pd.concat([salmon[[3]], trout[[3]]])

    c = GaussianNB()
    c.fit(x, y)
    print('8 - Bayesian Classifier + Whitening Transform - Error: ' +
          str(mean_squared_error(y, c.predict(x))))


def main():
    """ """
    print('  QUESTION 2:  ')
    q2c(plot_pdf=False)

    print('  QUESTION 3:  ')
    q3(plots=False)


if __name__ == "__main__":
    main()