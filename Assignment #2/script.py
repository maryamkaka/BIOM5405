import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def findOutliers(data):
    IQR = data.describe()['75%'] - data.describe()['25%']
    outliers = data[data > data.describe()['75%'] + 1.5*IQR]
    outliers.append(data[data<  data.describe()['25%'] + 1.5*IQR])
    return outliers

## Question 1 ##
print('\nQUESTION 1')
file = "assigData1.xls"
pci = pd.read_excel('assigData1.xls', 0)
psipred = pd.read_excel('assigData1.xls', 1)

print('(a) Spearman Rank')
print("PCI" + str(pci[["Q3", "Length"]].corr(method='spearman')))
print("PSIPRED" + str(psipred[["Q3", "Length"]].corr(method='spearman')))

## Question 2 ##
print('\nQUESTION 2')
file = 'assigData2.tsv'
data = pd.read_csv(file, sep='\t', index_col=False, header=None,
                names=["W_apl","W_orng", "W_grp", "D_apl", "D_orng", "D_grp"])
print('(a)')
print("Apple: " + str(stats.mstats.skewtest(data["W_apl"])))
print("Orange: " + str(stats.mstats.skewtest(data["W_orng"])))
print("Grape: " + str(stats.mstats.skewtest(data["W_grp"])))

print('(b)')
outliers = findOutliers(data.D_apl)
print("D_Apl outliers: \n" + str(outliers))
outliers = findOutliers(data.D_orng)
print("D_orng outliers: \n" + str(outliers))
outliers = findOutliers(data.D_grp)
print("D_grp outliers: \n" + str(outliers))

print("Original: " + str(data.D_grp.describe()))
print("No Outlier: " + str(data.D_grp[data.D_grp.index != outliers.index.values[0]].describe()))


print('(c)')
print('Min: ' + str(data['W_grp'].describe()['min']))
print('Max: ' + str(data['W_grp'].describe()['max']))
print('Range: ' + str(data['W_grp'].describe()['max'] - data['W_grp'].describe()['min']))
print('IQR: ' + str(data['W_grp'].describe()['75%'] - data['W_grp'].describe()['25%']))

## QUESTION 3 ##
print("\nQUESTION 3")

#create dataframe
col = ["0-5", "6-8", "9-10"]
rows = ["Underweight", "Normal Weight", "Overweight", "Obese"]
values = [[6, 4, 4], [2, 1, 2], [12, 6, 4], [4, 4, 1]]
df = (len(values) - 1) * (len(values[0]) - 1)
data = pd.DataFrame(values, index=rows, columns=col)
data = data.append(pd.Series(data.sum(), name="ColTotal"))
data = data.assign(RowTotal=pd.Series(data.sum(axis=1)))

#create contingency table
print('(a)')
contingencyTable = np.outer(data.RowTotal[0:-1], data.ix["ColTotal"][0:-1]) /data.ix["ColTotal","RowTotal"]
contingencyTable = pd.DataFrame(contingencyTable, index=rows, columns = col)
print(str(contingencyTable))
print('Degrees of Freedom: ' + str(df))

x, p = stats.chisquare(data.ix[0:-1, 0:-1], contingencyTable, ddof = df)
x = x.sum()
p = 1 - stats.chi2.cdf(x, df)
print("Chi^2: " + str(x))
print("p-value: " + str(p))
import pdb; pdb.set_trace()
