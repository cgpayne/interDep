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

import csv

# input and output file names
finSta = 'data/out_mrclean/states_PI.csv'
foutTmp = 'data/out_mrclean/states_PII.csv'


def uncsvip(filename):
    with open(filename, 'r', newline='\n') as fin:
        reader = csv.reader(fin)
        indata = list(reader)
    fin.close()
    lefty = []
    righty = []
    dlen = len(indata)
    for i in range(dlen):
        lefty.append(indata[i][0])
        righty.append(indata[i][1])
    return lefty, righty, dlen


# split all the words up, account for overlaps (set), then sort
def vocit(alist):
    return sorted(set(word for sentence in alist for word in sentence.split()))


liACH, liSta, stlen = uncsvip(finSta)

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
