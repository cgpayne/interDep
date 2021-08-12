#!/usr/bin/env python
#  mrcluster = cluster analysis of initial data
#  head -n 14 bigread.py
#  python3 mrcluster.py
#  By:  Charlie Payne
#  License: n/a
# DESCRIPTION
#  not sure yet
# NOTES
#  [none]
# KNOWN BUGS
#  [none]
# DESIRED FEATURES
#  [none]

import csv

# input file names
finSta = 'data/out_mrclean/states.csv'


with open(finSta, 'r', newline='\n') as fin:
    reader = csv.reader(fin)
    indata = list(reader)
fin.close()
liACH = []
liSta = []
dlen = len(indata)
for i in range(dlen):
    liACH.append(indata[i][0])
    liSta.append(indata[i][1])

# split all the words up, account for overlaps (set), then sort
vocabi = sorted(set(word for sentence in liSta for word in sentence.split()))
print(len(vocabi), vocabi)

for i in range(dlen):
    liSta[i] = liSta[i].replace('Basal ', 'Basal-')

vocabf = sorted(set(word for sentence in liSta for word in sentence.split()))
print(len(vocabf), vocabf)

print('FIN')
exit(0)
# FIN
