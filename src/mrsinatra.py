#!/usr/bin/env python
#  mrsinatra = initial data cleaning, PART II
#  head -n 14 mrsinatra.py
#  python3 mrsinatra.py
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
# import intdep_util as idu
from intdep_util import fclSt1, fsiSt2
from intdep_util import uncsvip

# input and output file names
# fclSt1 = 'data/out_mrclean/states_PI.csv'
foutTmp = 'data/out_mister/states_mrsinatra.csv'


def medcompound(alist):
    for i in range(len(alist)):
        # HMMM (lack of domain knowledge): Antigen, Cell, Clear,
        alist[i] = alist[i].replace('Basal ', 'Basal_')
        alist[i] = alist[i].replace('Non ', 'Non_')
        alist[i] = alist[i].replace('Soft ', 'Soft_')
        alist[i] = alist[i].replace(' Amp', '_Amp')  # HMMM....
        alist[i] = alist[i].replace('Central Nervous System', 'CNS')
        alist[i] = alist[i].replace('Peripheral Nervous System', 'PNS')
        alist[i] = alist[i].replace(' Grade', '_Grade')  # HMMM....
        alist[i] = alist[i].replace('Upper ', 'Upper_')
        alist[i] = alist[i].replace('Urinary Tract', 'Urinary_Tract')
    return alist


# split all the words up, account for overlaps (set), then sort
def vocit(alist):
    return sorted(set(word for sentence in alist for word in sentence.split()))


ykk, stlen = uncsvip(fclSt1)
liACH = ykk[0]
liOrg = ykk[1]
liCan = ykk[2]

vocabi = vocit(liCan)
print(len(vocabi), vocabi)

liOrg = medcompound(liOrg)
liCan = medcompound(liCan)

vocabf = vocit(liCan)
print(len(vocabf), vocabf)

with open(fsiSt2, 'w') as fout:
    for i in range(stlen):
        fout.write(liACH[i] + ',' + liOrg[i] + ',' + liCan[i] + '\n')
fout.close()


print('FIN')
exit(0)
# FIN
