import pandas as pd

## Question 1 ##
print('\nQUESTION 1')
file = "assigData1.xls"
pci = pd.read_excel('assigData1.xls', 0)
psipred = pd.read_excel('assigData1.xls', 1)

import pdb; pdb.set_trace()

print('(a) Spearman Rank')
print(pd.concat([pci.Q3, psipred.Q3], axis=1).corr(method='spearman'))
