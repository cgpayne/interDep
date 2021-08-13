#!/usr/bin/env python
#  mrcleanPII = initial data cleaning, PART II
#  head -n 14 mrcleanPII.py
#  python3 mrcleanPII.py
#  By:  Charlie Payne
#  License: n/a
# DESCRIPTION
#
# NOTES
#  [none]
# KNOWN BUGS
#  [none]
# DESIRED FEATURES
#  [none]

# import csv
import intdeputil as idu

# input and output file names
# idu.feaSt1 = 'data/out_mrclean/states_PI.csv'
foutTmp = 'data/out_mrclean/states_PII.csv'


# split all the words up, account for overlaps (set), then sort
def vocit(alist):
    return sorted(set(word for sentence in alist for word in sentence.split()))


ykk, stlen = idu.uncsvip(idu.feaSt1)
liACH = ykk[0]
liSta = ykk[1]

vocabi = vocit(liSta)
print(len(vocabi), vocabi)

for i in range(stlen):
    # HMMM (lack of domain knowledge): Antigen, Cell, Clear,
    liSta[i] = liSta[i].replace('Basal ', 'Basal_')
    liSta[i] = liSta[i].replace('Non ', 'Non_')
    liSta[i] = liSta[i].replace('Soft ', 'Soft_')
    liSta[i] = liSta[i].replace(' Amp', '_Amp')  # HMMM....
    liSta[i] = liSta[i].replace('Central Nervous System', 'CNS')
    liSta[i] = liSta[i].replace('Peripheral Nervous System', 'PNS')
    liSta[i] = liSta[i].replace(' Grade', '_Grade')  # HMMM....
    liSta[i] = liSta[i].replace('Upper ', 'Upper_')
    liSta[i] = liSta[i].replace('Urinary Tract', 'Urinary_Tract')

vocabf = vocit(liSta)
print(len(vocabf), vocabf)

with open(foutTmp, 'w') as fout:
    for i in range(stlen):
        fout.write(liACH[i] + ',' + liSta[i] + '\n')


print('FIN')
exit(0)
# FIN
