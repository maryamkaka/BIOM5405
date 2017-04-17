import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import precision_recall_curve, mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics.ranking import _binary_clf_curve

OUTPUT_LOCATION = 'images/'


def modified_precision_recall_curve(r, y_true, probas_pred, pos_label=None,
                                    sample_weight=None):
    """ Precision Recall Calculations using Prevalence-corrected precision 
        Code modified from sklearn.metrics.precision_recall_curve
    """

    """fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).

    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps)."""
    fps, tps, thresholds = _binary_clf_curve(y_true, probas_pred,
                                             pos_label=pos_label,
                                             sample_weight=sample_weight)
    recall = tps / tps[-1]
    sp = (fps[-1] - fps)/(fps[-1])
    precision = recall / (recall + (r*(1-sp)))

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def pr(r, n=1000):
    pos = np.random.normal(50, 5, n)
    neg = np.random.normal(60, 12, n*r)

    precision, recall, thresh = modified_precision_recall_curve(r,
        np.concatenate((np.ones(pos.shape), np.zeros(neg.shape))),
        np.concatenate((pos, neg))
    )

    return precision, recall


def q2c(n=1000):
    """ 
    Prevalence-corrected precision is defined as 𝑝𝑐𝑃𝑅 = 𝑆𝑛/(𝑆𝑛+𝑟(1−𝑆𝑝)) 
    where r reflects the number of negative samples for each positive sample. 
    This is useful when you have a limited number of test samples even though 
    the expected true class imbalance may be very high (e.g. 1000 negatives for 
    each positive). Assume you have a classifier which outputs scores for test 
    samples. Simulate classification scores for 1000 positive test samples 
    distributed as N(50,5) and 1000 negative test sample scores distributed as
    N(60,12).
    
    :parameter:
     n: number of samples; Integer; Default: 1000
    
    """
    pcPR = lambda sn, r, sp: sn/(sn + r*(1-sp))
    pos = np.random.normal(50, 5, n)
    neg = np.random.normal(60, 12, n)

    pos_dist = stats.norm(loc=50, scale=5)
    neg_dist = stats.norm(loc=60, scale=12)
    x = np.linspace(0, 100, 1000)
    plt.figure()
    plt.title('PDF of Class Distributions')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.plot(x, pos_dist.pdf(x))
    plt.plot(x, neg_dist.pdf(x))
    if plot_pdf: plt.savefig('2ci.png', bbox_inches='tight')

    precision, recall, thresh = precision_recall_curve(
        np.concatenate((np.ones(pos.shape), np.zeros(neg.shape))),
        np.concatenate((pos, neg))
    )

    p10, r10 = pr(10)
    p100, r100 = pr(100)
    p1000, r1000 = pr(1000)

    plt.figure()
    plt.title('Precision Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.plot(recall[1:-1], precision[1:-1], label='1:1')
    plt.plot(r10[1:-1], p10[1:-1], label='10:1')
    plt.plot(r100[1:-1], p100[1:-1], label='100:1')
    plt.plot(r1000[1:-1], p1000[1:-1], label='1000:1')
    plt.legend(loc='upper left')
    plt.savefig(OUTPUT_LOCATION + '2ciii.png', bbox_inches='tight')


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

    # visualize original data
    if plots:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(salmon[0], salmon[1], salmon[2], label='Salmon')
        ax.scatter(trout[0], trout[1], trout[2], c='b', marker='^',
                   label='Trout')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        ax.legend(loc='upper left')

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
        plt.show()

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
    x = np.concatenate([normalize(salmon[[0, 1, 2]]),
                        normalize(trout[[0, 1, 2]])])
    y = pd.concat([salmon[[3]], trout[[3]]])

    c = GaussianNB()
    c.fit(x[:, (1, 2)], y)
    print('6 - Bayesian Classifier + Normalization + Remove F1 - Error: ' +
          str(mean_squared_error(y, c.predict(x[:, (1, 2)]))))

    c = GaussianNB()
    c.fit(x[:, (0, 1)], y)
    print('7 - Bayesian Classifier + Normalization + Remove F3 - Error: ' +
          str(mean_squared_error(y, c.predict(x[:, (0, 1)]))))
    if plots:
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.scatter(normalize(salmon[0]), normalize(salmon[1]), label='Salmon')
        plt.scatter(normalize(trout[0]), normalize(trout[1]), c='b', marker='^',
                    label='Trout')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True)
        plt.subplot(1, 3, 2)
        plt.title('Plot of Normalized Data Points')
        plt.scatter(normalize(salmon[0]), normalize(salmon[2]), label='Salmon')
        plt.scatter(normalize(trout[0]), normalize(trout[2]), c='b', marker='^',
                    label='Trout')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 3')
        plt.grid(True)
        plt.subplot(1, 3, 3)
        plt.scatter(normalize(salmon[1]), normalize(salmon[2]), label='Salmon')
        plt.scatter(normalize(trout[1]), normalize(trout[2]), c='b', marker='^',
                    label='Trout')
        plt.xlabel('Feature 2')
        plt.ylabel('Feature 3')
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.show()

    # Bayesian Classifier + Whitening Transform
    salmon_t = PCA().fit_transform(salmon[[0, 1, 2]])
    trout_t = PCA().fit_transform(trout[[0, 1, 2]])

    if plots:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(salmon_t[:,0], salmon_t[:,1], salmon_t[:,2], label='Salmon')
        ax.scatter(trout_t[:, 0], trout_t[:, 1], trout_t[:, 2], c='b',
                   marker='^', label='Trout')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        ax.legend(loc='upper left')
        plt.show()

    x = np.concatenate((salmon_t, trout_t))
    y = pd.concat([salmon[[3]], trout[[3]]])

    c = GaussianNB()
    c.fit(x, y)
    print('8 - Bayesian Classifier + Whitening Transform - Error: ' +
          str(mean_squared_error(y, c.predict(x))))


def main():
    """ """
    print('--- QUESTION 2 ---')
    q2c()

    print('--- QUESTION 3 ---')
    q3()


if __name__ == "__main__":
    main()