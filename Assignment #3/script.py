import pandas as pd
import numpy as np
import scipy.stats as stats
import math
import matplotlib.pyplot as plt
import bisect
from sklearn import metrics
from sklearn import utils

outputLocation = 'Assignment #3/images/'

def chiSquared(data):
    df = ((data.shape[0] - 1) - 1) * ((data.shape[1] - 1) - 1)
    contingencyTable = np.outer(data.RowTotal[0:-1], data.ix["ColTotal"][0:-1])\
        /data.ix["ColTotal","RowTotal"]
    contingencyTable = pd.DataFrame(contingencyTable, index=rows, columns = col)

    x, p = stats.chisquare(data.ix[0:-1, 0:-1], contingencyTable, ddof = df)
    x = x.sum()
    p = 1 - stats.chi2.cdf(x, df)

    print('Degrees of Freedom: ' + str(df))
    print("Chi^2: " + str(x))
    print("p-value: " + str(p))

def confusionMatrix(tp, fn, fp, tn):
    data = pd.DataFrame([[tp, fn], [fp, tn]], index=rows, columns=col)
    data = data.append(pd.Series(data.sum(), name="ColTotal"))
    data = data.assign(RowTotal=pd.Series(data.sum(axis=1)))
    print(data)

    return data

def bootstrap(data, nSamples):
    stats = np.empty(nSamples)
    for i in range(0, nSamples):
        sample = utils.resample(data)
        stats[i] = findPrecision(sample['class'], sample['score'])
    np.sort(stats)
    return [stats[math.floor(nSamples*0.025)], stats[math.ceil(nSamples*0.975)]]

def findPrecision(c, s):
    precision, recall, t = metrics.precision_recall_curve(c, s, pos_label=1)
    points = np.array([[i, x] for i,x in enumerate(recall) if x >= 0.75])
    points = points[np.argsort(points[:, 1])]
    return precision[points[0][0]]

print("QUESTION 1")
col = ["TestT", "TestF"]
rows = ["ActualT", "ActualF"]

SENS = 0.65
SPEC = 0.62

ppv = lambda tp, fp: tp/(tp+fp)

print('(i)')
print('a.')
nPos = 100; nNeg = 100
tp = SENS*nPos; fn = nPos - tp; tn = SPEC*nNeg; fp = nNeg - tn
PPVa = ppv(tp, fp)
chiSquared(confusionMatrix(tp, fn, fp, tn))

print('\nb.')
nPos = 100; nNeg = 1000
tp = SENS*nPos; fn = nPos - tp; tn = SPEC*nNeg; fp = nNeg - tn
PPVb = ppv(tp, fp)
chiSquared(confusionMatrix(tp, fn, fp, tn))

print('\nc.')
nPos = 400; nNeg = 400
tp = SENS*nPos; fn = nPos - tp; tn = SPEC*nNeg; fp = nNeg - tn
PPVc = ppv(tp, fp)
chiSquared(confusionMatrix(tp, fn, fp, tn))

print('\n(ii)')
print('a. PPV: ' + str(PPVa))
print('b. PPV: ' + str(PPVb))
print('c. PPV: ' + str(PPVc))

plt.figure()
plt.plot(0.5, 0.5, marker='P', label='Random Classifier')
plt.plot(SENS, PPVa, marker='o', label='a')
plt.plot(SENS, PPVb, marker='s', label='b')
plt.plot(SENS, PPVc, marker='x', label='c')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title('Precision Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc = 'lower left')
plt.grid(True)
plt.savefig(outputLocation + 'precision-recall.png', bbox_inches='tight')

print('\nQUESTION 2')
data = pd.read_csv("Assignment #3/assigData3.tsv", sep='\t', index_col=False,
    header=None, names=["score", "class"])

print('(i)')
fpr, tpr, thresh = metrics.roc_curve(data['class'], data['score'], pos_label=1)
auc = metrics.auc(fpr, tpr)

print('AUC: ' + str(auc))

plt.figure()
plt.plot(fpr, tpr, label='ROC Curve (AUC = %0.2f)' %auc)
plt.title('Reciever Operating Charateristic Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.grid(True)
plt.savefig(outputLocation + 'ROC.png', bbox_inches='tight')

print('(ii)')
minSens = 0.75
index = bisect.bisect_left(tpr, minSens)
print('Max Specificity: ' + str(1-fpr[index]))

print('(iii)')
precision, recall, t = metrics.precision_recall_curve(data['class'],
    data['score'], pos_label=1)
auc = metrics.auc(recall[1:-1], precision[1:-1])

print('(iv)')
index = [i for i,x in enumerate(recall) if x == 0.75][0]
print('Precision (@75'+ '%' +' Sensitivity): ' + str(findPrecision(data['class'],
    data['score'])))

plt.figure()
plt.plot(recall[1:-1], precision[1:-1],
    label='Precision-Recall Curve (AUC = %0.2f)' %auc)
plt.plot(recall[index], precision[index], marker='o',
    label='75% Sensitivity Point')
plt.title('Precision Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc = 'lower right')
plt.grid(True)
plt.savefig(outputLocation + 'precision-recall2.png', bbox_inches='tight')

print('(v)')
print(bootstrap(data, 1000))
