import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

## Question 1 ##
print('\nQUESTION 1')
file = "assigData1.xls"
pci = pd.read_excel('assigData1.xls', 0)
psipred = pd.read_excel('assigData1.xls', 1)

print('(a) Spearman Rank')
print(pd.concat([pci.Q3, psipred.Q3], axis=1).corr(method='spearman'))

print('\nQUESTION 2')
file = 'assigData2.tsv'
data = pd.read_csv(file, sep='\t', index_col=False, header=None,
                names=["W_apl","W_orng", "W_grp", "D_apl", "D_orng", "D_grp"])
print('(a)')
print("Apple: " + str(stats.mstats.skewtest(data["W_apl"])))
print("Orange: " + str(stats.mstats.skewtest(data["W_orng"])))
print("Grape: " + str(stats.mstats.skewtest(data["W_grp"])))

print('(b)')

print('(c)')
print('Min: ' + str(data['W_grp'].describe()['min']))
print('Max: ' + str(data['W_grp'].describe()['max']))
print('Range: ' + str(data['W_grp'].describe()['max'] - data['W_grp'].describe()['min']))
print('IQR: ' + str(data['W_grp'].describe()['75%'] - data['W_grp'].describe()['25%']))
