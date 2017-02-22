import pandas as pd
import numpy
import matplotlib.pyplot as plt
import bisect
from sklearn import metrics

outputLocation = 'Assignment #3/images/'

print("QUESTION 1")
SENS = 0.65
SPEC = 0.62
nPos = 100
nNeg = 100
tp = SENS
fn = nPos - SENS
fp = nNeg - SPEC
tn = SPEC

print('QUESTION 2')
data = pd.read_csv("Assignment #3/assigData3.tsv", sep='\t', index_col=False, header=None,
    names=["score", "class"])

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
precision, recall, t = metrics.precision_recall_curve(data['class'], data['score'], pos_label=1)
auc = metrics.auc(recall[1:-1], precision[1:-1])

plt.figure()
plt.plot(recall[1:-1], precision[1:-1], label='Precision-Recall Curve (AUC = %0.2f)' %auc)
plt.title('Precision Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc = 'lower right')
plt.grid(True)
plt.savefig(outputLocation + 'precision-recall.png', bbox_inches='tight')

print('(v)')
index = index = [i for i,x in enumerate(recall) if x == 0.75][0]
print('Precision (@75'+ '%' +' Sensitivity): ' + str(precision[index]))
