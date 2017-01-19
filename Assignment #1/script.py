import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import error_ellipse as ellipse

#variable initialization
outputLocation = 'images/'

def question1(className, data):
    # Part a
    plt.figure()
    plt.scatter(data["Length"], data["Q3"])
    plt.title('Question 1a - Plot of Q3 Accuracy vs. Protein Length - ' +
        className)
    plt.ylabel('Q3 Accuracy')
    plt.xlabel('Protein Length')
    plt.grid(True)
    plt.savefig(outputLocation + 'Question1a-' + className + '.png',
        bbox_inches='tight')

    # Part b
    print("\n(b) Correlation Calculation for " + className)
    print(data[["Q3", "Length"]].corr()) #default Pearson Correlation

    # Part c
    print("\n(c) Statistical Summary for " + className)
    print(data.describe())

def plotHist(data, xlabel, ylabel, nbin):
    plt.figure()
    data.plot.hist(bins=nbin, histtype='step')
    plt.title('Histogram of ' + xlabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(outputLocation + xlabel + '-bin' + str(nbin) + '.png',
        bbox_inches='tight')

def main():
    ## Question 1 ##
    print('\nQUESTION 1')
    file = "assigData1.xls"
    question1('PCI', pd.read_excel('assigData1.xls', 0))
    question1('PCIPRED', pd.read_excel('assigData1.xls', 1))

    ## Question 2 ##
    print('\nQUESTION 2')
    file = 'assigData2.tsv'
    data = pd.read_csv(file, sep='\t', index_col=False, header=None,
        names=["W_apl","W_orng", "W_grp", "D_apl", "D_orng", "D_grp"])

    # part a
    print('\n(a) Maximum Likelihood Estimator')
    print('Mean: \n' + str(data.mean()))
    print('Variance: \n' + str(data.var(ddof=0)))   #default is normalized to N-1

    # part b
    plotHist(data[["W_apl", "W_orng", "W_grp"]], 'Weight', 'Frequency', 10)
    plotHist(data[["D_apl", "D_orng", "D_grp"]], 'Diameters', 'Frequency', 10)
    plotHist(data[["W_apl", "W_orng", "W_grp"]], 'Weight', 'Frequency', 25)
    plotHist(data[["D_apl", "D_orng", "D_grp"]], 'Diameters', 'Frequency', 25)
    plotHist(data[["W_apl", "W_orng", "W_grp"]], 'Weight', 'Frequency', 50)
    plotHist(data[["D_apl", "D_orng", "D_grp"]], 'Diameters', 'Frequency', 50)

    # part c
    combined = pd.concat([data["W_apl"], data["W_orng"], data["W_grp"]])
    combined = combined.apply(np.floor)
    plotHist(combined, 'Combined Weight Data', 'Frequency', 15)
    k, p = stats.mstats.normaltest(combined)
    print("\n(c) Normal Test: " + str(k) + " (p-value: " + str(p) + ")")

    ## Question 3 ##
    print('\nQUESTION 3')

    # part a
    mu = [3.2, 5.1]
    cov = np.array([[1.2, -0.5], [-0.5, 3.3]])
    n = 1000
    points = np.random.multivariate_normal(mu, cov, n)
    x, y = points.T

    # part b
    plt.figure()
    plt.plot(x, y, 'b.')
    plt.axis('equal')
    plt.grid(True)
    plt.title('Scatter plot of bivariate normal distribution')
    plt.savefig(outputLocation+'Question3b.png', bbox_inches='tight')

    # part c
    print('\n(c)')
    print('Det: ' + str(np.linalg.det(cov)))
    print('Trace: ' + str(np.matrix.trace(cov)))

    # part d
    print('\n(d)')
    eigenvals, eigenvect = np.linalg.eig(cov)
    print('eigenvalues: ' + str(eigenvals))
    print('eigenvectors: \n' + str(eigenvect))
    ellipse.plot_point_cov(points, nstd=2, color='k', alpha=0.5)
    plt.axis('equal')
    plt.grid(True)
    plt.title('Ellipse of equiprobability')
    plt.savefig(outputLocation + 'ellipse.png', bbox_inches='tight')

    # part e
    mu = 3.2
    var = 1.2
    std = math.sqrt(var)
    dist = stats.norm(loc=mu, scale=var)
    x = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 100)
    plt.figure()
    plt.title('PDF of 1D Normal Distribuition (mu = ' + str(mu) + ', var = ' +
        str(var) + ')')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.plot(x, dist.pdf(x))
    plt.savefig(outputLocation + 'pdf.png', bbox_inches='tight')
    plt.figure()
    plt.title('CDF of 1D Normal Distribuition (mu = ' + str(mu) + ', var = ' +
        str(var) + ')')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.plot(x, dist.cdf(x))
    plt.savefig(outputLocation + 'cdf.png', bbox_inches='tight')
main();
